"""
Adversarial intra-processing algorithm by Savani et al. (2020) [https://arxiv.org/abs/2006.08564].

Code adapted from https://github.com/abacusai/intraprocessing_debiasing
"""
import copy

import math

import numpy as np

import torch
from torch import optim, nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from models.networks_tabular import load_model, Critic as TabularCritic
from utils.evaluation import (
    eval_model_w_data_loaders,
    get_valid_objective_,
    get_test_objective,
    get_valid_objective,
    get_test_objective_,
    compute_empirical_bias,
    get_best_thresh
)
from models.networks_ChestXRay import Critic as CXR_Critic
from utils.logging_utils import get_logger

logger = get_logger("Adversarial Model Debiasing")

def evaluate_adversarial_model(model, dataloaders, dataset_sizes, config, device):
    """
    Perform adversarial in-processing evaluation.
    Trains critic and actor iteratively, finds best threshold, then evaluates on validation and test sets.
    Returns dicts keyed by 'adversarial' for validation and test results.
    """
    # Prepare adversarial components
    batch_size = config['adversarial']['batch_size']
    # Base feature extractor from model (assumes vgg16 attr)
    base_model = copy.deepcopy(model.vgg16)
    base_model.classifier[-1] = nn.Linear(
        base_model.classifier[-1].in_features,
        base_model.classifier[-1].in_features
    )
    actor = nn.Sequential(
        base_model,
        nn.Linear(base_model.classifier[-1].in_features, 2)
    ).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config['adversarial']['lr'])
    actor_loss_fn = nn.BCEWithLogitsLoss()

    critic = CXR_Critic(
        config['adversarial']['batch_size'] * base_model.classifier[-1].in_features
    ).to(device)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config['adversarial'].get('critic_lr', 1e-4))
    critic_loss_fn = nn.MSELoss()

    actor_steps = config['adversarial']['actor_steps']
    critic_steps = config['adversarial']['critic_steps']
    epochs = config['adversarial']['epochs']

    scaler_actor = GradScaler()
    scaler_critic = GradScaler()

    # Iterative training
    logger.info("Starting adversarial training for %d epochs.", epochs)
    for epoch in range(epochs):
        logger.info("Epoch %d/%d", epoch + 1, epochs)
        # Train critic
        critic.train()
        actor.eval()
        critic_bar = tqdm(enumerate(dataloaders['val']), total=critic_steps, desc=f"Critic Epoch {epoch+1}", leave=False)
        for step, (X, y, p) in critic_bar:
            
            if step >= critic_steps:
                break
            X, y, p = X.to(device), y.to(device), p.to(device)
            
            if X.size(0) != batch_size:
                continue
            critic_optimizer.zero_grad()
            
            with torch.no_grad(), autocast(device_type=device.type):
                y_pred = actor(X)
            
            bias = compute_empirical_bias(y_pred, y.float(), p.float(), config['metric'])
            
            with autocast(device_type=device.type):
                res = critic(base_model(X))
                loss = critic_loss_fn(bias.unsqueeze(0), res[0])
            
            scaler_critic.scale(loss).backward()
            scaler_critic.step(critic_optimizer)
            scaler_critic.update()
            critic_bar.set_postfix(loss=loss.item())

        # Train actor
        critic.eval()
        actor.train()
        actor_bar = tqdm(enumerate(dataloaders['val']), total=actor_steps, desc=f"Actor Epoch {epoch+1}", leave=False)
        for step, (X, y, p) in actor_bar:
            if step >= actor_steps:
                break
            X, y, p = X.to(device), y.to(device), p.to(device)
            if X.size(0) != batch_size:
                continue
            actor_optimizer.zero_grad()
            with autocast(device_type=device.type):
                est_bias = critic(base_model(X))
                loss = actor_loss_fn(actor(X)[:, 0], y.float())
                # scale loss by bias constraint
                margin = config['adversarial']['margin']
                epsilon = config['objective']['epsilon']
                scaled = max(1, config['adversarial']['lambda'] * (abs(est_bias) - epsilon + margin) + 1)
                loss = loss * scaled
            scaler_actor.scale(loss).backward()
            scaler_actor.step(actor_optimizer)
            scaler_actor.update()
            actor_bar.set_postfix(loss=loss.item())

    # Determine threshold via validation
    logger.info("Training completed. Determining best threshold.")
    _, best_thresh = val_model_dataloaders(
        actor, dataloaders['val'], get_best_objective, device, config
    )
    best_thresh = best_thresh.cpu().numpy()
    logger.info("Determined adversarial best threshold: %.4f", best_thresh)

    # Evaluate final actor
    logger.info("Starting final evaluation.")
    actor.eval()
    with torch.no_grad(), autocast(device_type=device.type):
        valid_scores, y_valid, p_valid = eval_model_w_data_loaders(
            model=actor,
            device=device,
            dataloader=dataloaders['val'],
            dataset_size=dataset_sizes['val'],
            batch_size=batch_size
        )
    
    with torch.no_grad(), autocast(device_type=device.type):
        test_scores, y_test, p_test = eval_model_w_data_loaders(
            model=actor,
            device=device,
            dataloader=dataloaders['test'],
            dataset_size=dataset_sizes['test'],
            batch_size=batch_size
        )

    results_valid = {
        'adversarial': get_valid_objective_(
            y_pred=(valid_scores > best_thresh),
            y_val=y_valid,
            p_val=p_valid,
            config=config
        )
    }
    results_test = {
        'adversarial': get_test_objective_(
            y_pred=(test_scores > best_thresh),
            y_test=y_test,
            p_test=p_test,
            config=config
        )
    }
    logger.info("Adversarial validation results: %s", results_valid['adversarial'])
    logger.info("Adversarial test results: %s", results_test['adversarial'])

    # Cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    logger.info("GPU cache cleared.")

    return results_valid, results_test



def val_model_dataloaders(model, loader, criterion, device, config):
    """Validate model on loader with criterion function"""
    y_true, y_pred, y_prot = [], [], []
    model.eval()
    with torch.no_grad():
        for X, y, p in loader:
            X, y, p = X.to(device), y.float().to(device), p.float().to(device)
            y_true.append(y)
            y_prot.append(p)
            y_pred.append(torch.sigmoid(model(X)[:, 0]))
    y_true, y_pred, y_prot = torch.cat(y_true), torch.cat(y_pred), torch.cat(y_prot)
    return criterion(y_true, y_pred, y_prot, config)


def get_best_objective(y_true, y_pred, y_prot, config):
    """Find the threshold for the best objective"""
    num_samples = 5
    threshs = torch.linspace(0, 1, 101)
    best_obj, best_thresh = -math.inf, 0.
    for thresh in threshs:
        indices = np.random.choice(np.arange(y_pred.size()[0]), num_samples*y_pred.size()[0],
                                   replace=True).reshape(num_samples, y_pred.size()[0])
        objs = []
        for index in indices:
            y_pred_tmp = y_pred[index]
            y_true_tmp = y_true[index]
            y_prot_tmp = y_prot[index]
            perf = (torch.mean((y_pred_tmp > thresh)[y_true_tmp.type(torch.bool)].type(torch.float32)) +
                    torch.mean((y_pred_tmp <= thresh)[~y_true_tmp.type(torch.bool)].type(torch.float32))) / 2
            bias = compute_empirical_bias((y_pred_tmp > thresh).float().cpu(), y_true_tmp.float().cpu(),
                                          y_prot_tmp.float().cpu(), config['metric'])
            objs.append(compute_objective(perf, bias))
        obj = float(torch.tensor(objs).mean())
        if obj > best_obj:
            best_obj, best_thresh = obj, thresh

    return best_obj, best_thresh


def compute_objective(performance, bias, epsilon=0.05, margin=0.01):
    """Evaluate constrained objective"""
    if abs(bias) <= (epsilon-margin):
        return performance
    else:
        return 0.0


def adversarial_debiasing(model_state_dict, data, config, device):
    """Runs adversarial debiasing on the given trained model and the validation set."""
    logger.info('Training Adversarial model.')
    actor = load_model(data.num_features, config.get('hyperparameters', {}))
    actor.load_state_dict(model_state_dict)
    actor.to(device)
    hid = config['hyperparameters']['hid'] if 'hyperparameters' in config else 32
    critic = TabularCritic(hid * config['adversarial']['batch_size'], num_deep=config['adversarial']['num_deep'], hid=hid)
    critic.to(device)
    critic_optimizer = optim.Adam(critic.parameters())
    critic_loss_fn = torch.nn.MSELoss()

    actor_optimizer = optim.Adam(actor.parameters(), lr=config['adversarial']['lr'])
    actor_loss_fn = torch.nn.BCELoss()

    for epoch in range(config['adversarial']['epochs']):
        for param in critic.parameters():
            param.requires_grad = True
        for param in actor.parameters():
            param.requires_grad = False
        actor.eval()
        critic.train()
        for step in range(config['adversarial']['critic_steps']):
            critic_optimizer.zero_grad()
            indices = torch.randint(0, data.X_valid.size(0), (config['adversarial']['batch_size'],))
            cX_valid = data.X_valid_gpu[indices]
            cy_valid = data.y_valid[indices]
            cp_valid = data.p_valid[indices]
            with torch.no_grad():
                scores = actor(cX_valid)[:, 0].reshape(-1).cpu().numpy()

            bias = compute_empirical_bias(scores, cy_valid.numpy(), cp_valid, config['metric'])

            res = critic(actor.trunc_forward(cX_valid))
            loss = critic_loss_fn(torch.tensor([bias], device=device).float(), res[0])
            loss.backward()
            train_loss = loss.item()
            critic_optimizer.step()
            if (epoch % 10 == 0) and (step % 100 == 0):
                logger.info(f'=======> Critic Epoch: {(epoch, step)} loss: {train_loss}')

        for param in critic.parameters():
            param.requires_grad = False
        for param in actor.parameters():
            param.requires_grad = True
        actor.train()
        critic.eval()
        for step in range(config['adversarial']['actor_steps']):
            actor_optimizer.zero_grad()
            indices = torch.randint(0, data.X_valid.size(0), (config['adversarial']['batch_size'],))
            cy_valid = data.y_valid_gpu[indices]
            cX_valid = data.X_valid_gpu[indices]

            pred_bias = critic(actor.trunc_forward(cX_valid))
            bceloss = actor_loss_fn(actor(cX_valid)[:, 0], cy_valid)

            objloss = max(
                1, config['adversarial']['lambda'] * (abs(pred_bias[0][0]) - config['objective']['epsilon'] +
                                                      config['adversarial']['margin']) + 1) * bceloss

            objloss.backward()
            train_loss = objloss.item()
            actor_optimizer.step()
            if (epoch % 10 == 0) and (step % 100 == 0):
                logger.info(f'=======> Actor Epoch: {(epoch, step)} loss: {train_loss}')

        if epoch % 1 == 0:
            with torch.no_grad():
                scores = actor(data.X_valid_gpu)[:, 0].reshape(-1, 1).cpu().numpy()
                _, best_adv_obj = get_best_thresh(scores, np.linspace(0, 1, 101), data, config,
                                                  margin=config['adversarial']['margin'])
                logger.info(f'Objective: {best_adv_obj}')

    logger.info('Finding optimal threshold for Adversarial model.')
    with torch.no_grad():
        scores = actor(data.X_valid_gpu)[:, 0].reshape(-1, 1).cpu().numpy()

    best_adv_thresh, _ = get_best_thresh(scores, np.linspace(0, 1, 101), data, config,
                                         margin=config['adversarial']['margin'])

    logger.info('Evaluating Adversarial model on best threshold.')
    with torch.no_grad():
        labels = (actor(data.X_valid_gpu)[:, 0] > best_adv_thresh).reshape(-1, 1).cpu().numpy()
    results_valid = get_valid_objective(labels, data, config)
    logger.info(f'Results: {results_valid}')

    with torch.no_grad():
        labels = (actor(data.X_test_gpu)[:, 0] > best_adv_thresh).reshape(-1, 1).cpu().numpy()
    results_test = get_test_objective(labels, data, config)

    return results_valid, results_test
