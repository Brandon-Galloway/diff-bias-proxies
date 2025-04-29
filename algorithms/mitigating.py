"""
Adversarial in-processing algorithm by Zhang et al. (2018) [DOI: https://doi.org/10.1145/3278721.3278779]

Code adapted from https://github.com/choprashweta/Adversarial-Debiasing and
																https://github.com/abacusai/intraprocessing_debiasing
"""
import time

import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.amp import autocast, GradScaler

from tqdm import tqdm

from models.networks_ChestXRay import (ChestXRayVGG16, ChestXRayResNet18)

from utils.evaluation import (
    compute_empirical_bias,
    compute_accuracy_metrics,
    eval_model_w_data_loaders,
    find_best_threshold,
    get_valid_objective_,
    get_test_objective_
)

from models.networks_tabular import load_model
from utils.evaluation import get_valid_objective, get_test_objective, get_best_thresh
from utils.logging_utils import get_logger

logger = get_logger("Mitigating Model Debiasing")


def evaluate_mitigating_model(model, dataloaders, dataset_sizes, config, device):
    """
    Perform in-processing mitigation via bias-gradient descent/ascent (train_fair_ChestXRay_model).
    Returns dicts keyed by 'mitigating' for validation and test results.
    """
    batch_size = config['mitigating']['batch_size']
    epochs = config['mitigating']['n_epochs']

    logger.info("Starting in-processing mitigation training for %d epochs.", epochs)
    # Train a debiased model
    mitigated_model, *_ = train_fair_ChestXRay_model(
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        config=config,
        bias_metric='eod',
        batch_size=batch_size,
        num_epochs=epochs
    )
    mitigated_model.to(device)
    mitigated_model.eval()

    # Evaluate on validation and test
    logger.info("Evaluating mitigated model.")
    with torch.no_grad(), autocast(device_type=device.type):
        valid_scores, y_valid, p_valid = eval_model_w_data_loaders(
            model=mitigated_model,
            device=device,
            dataloader=dataloaders['val'],
            dataset_size=dataset_sizes['val'],
            batch_size=batch_size
        )
    
    with torch.no_grad(), autocast(device_type=device.type):
        test_scores, y_test, p_test = eval_model_w_data_loaders(
            model=mitigated_model,
            device=device,
            dataloader=dataloaders['test'],
            dataset_size=dataset_sizes['test'],
            batch_size=batch_size
        )

    logger.info("Finding best threshold for debiased model.")
    best_thresh = find_best_threshold(
        valid_scores, y_valid, config['acc_metric'])
    logger.info("Best threshold for mitigating: %.4f", best_thresh)

    # Compute objectives
    valid_obj = get_valid_objective_(
        y_pred=(valid_scores > best_thresh),
        y_val=y_valid,
        p_val=p_valid,
        config=config
    )
    logger.info("Validation results: %s", valid_obj)

    test_obj = get_test_objective_(
        y_pred=(test_scores > best_thresh),
        y_test=y_test,
        p_test=p_test,
        config=config
    )
    logger.info("Test results: %s", test_obj)
    
    # Cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    return {'mitigating': valid_obj}, {'mitigating': test_obj}


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()

        self.arch = config['default']['arch']

        if config['default']['arch'] == 'vgg':
            self.base_model = ChestXRayVGG16(
                pretrained=config['default']['pretrained'])
        elif config['default']['arch'] == 'resnet':
            self.base_model = ChestXRayResNet18(
                pretrained=config['default']['pretrained'])
        else:
            NotImplementedError('Architecture not supported!')

    def forward(self, t, training_mode=False):
        if self.arch == 'vgg':
            classifier_prev_output = torch.relu(self.base_model.vgg16(t))
        elif self.arch == 'resnet':
            classifier_prev_output = torch.relu(self.base_model.resnet18(t))

        classifier_output = self.base_model.out(classifier_prev_output)
        
        return (classifier_output, classifier_prev_output) if training_mode else torch.sigmoid(classifier_output)

class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()

        self.a1 = nn.Linear(1000, 512)
        self.a2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.a1(x))
        return self.a2(x)


def train_adversary(adv, clf, optimizer_adv, train_loader, loss_criterion, device, n_steps):
    adv.train()
    steps = 0

    bar = tqdm(train_loader, total=n_steps, desc="Training Adversary", leave=False)
    for inputs, labels, attrs in bar:
        if steps > n_steps:
            break
        # get the inputs and labels
        inputs, labels, attrs = inputs.to(device), labels.to(device), attrs.to(device)

        optimizer_adv.zero_grad()

        classifier_output, classifier_prev_output = clf(inputs, training_mode=True)
        
        adversary_output = adv(classifier_prev_output)
        # NOTE: this adversarial loss is only applicable for the EOD (see the original paper by Zhang et al.)
        loss = loss_criterion(torch.sigmoid(adversary_output[labels == 1])[:, 0], attrs[labels == 1].float())
        loss.backward()
        optimizer_adv.step()
        bar.set_postfix(loss=loss.item())
        steps += 1

    return adv


def train_classifier(clf, optimizer_clf, adv, train_loader, loss_criterion, lbda, device, n_steps):
    clf.train()
    steps = 0
    bar = tqdm(train_loader, total=n_steps, desc="Training Classifier", leave=False)

    for inputs, labels, attrs in bar:
        if steps > n_steps:
            break
        # get the inputs and labels
        inputs, labels, attrs = inputs.to(device), labels.to(device), attrs.to(device)

        optimizer_clf.zero_grad()

        classifier_output, classifier_prev_output = clf(inputs, training_mode=True)
        adversary_output = adv(classifier_prev_output)
        
        classifier_loss = loss_criterion(torch.sigmoid(classifier_output)[:, 0], labels.float())
        adversary_loss = loss_criterion(torch.sigmoid(adversary_output[labels == 1])[:, 0], attrs[labels == 1].float())
        total_loss = classifier_loss - lbda * adversary_loss
        total_loss.backward()
        optimizer_clf.step()
        bar.set_postfix(loss=total_loss.item())
        steps += 1
        
    return clf


def train_fair_ChestXRay_model(dataloaders, dataset_sizes, device, config, bias_metric='spd', batch_size=256,
                               num_epochs=25):
    since = time.time()

    model = Classifier(config=config)
    adversary = Adversary()

    model.to(device)
    adversary.to(device)

    optimizer_clf = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-8)
    optimizer_adv = optim.AdamW(
        adversary.parameters(), lr=1e-4, weight_decay=1e-8)

    loss_criterion_clf = nn.BCEWithLogitsLoss()
    loss_criterion_adv = nn.BCEWithLogitsLoss()

    # Best model parameters
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf
    best_acc = 0
    best_bias = 0
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    train_bias = []
    val_bias = []

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Iterate over epochs
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                adversary.train()

                for param in model.parameters():
                    param.requires_grad = False
                adversary = train_adversary(adv=adversary, clf=model, optimizer_adv=optimizer_adv,
                                            train_loader=dataloaders['train'], loss_criterion=loss_criterion_adv,
                                            device=device, n_steps=100)

                for param in model.parameters():
                    param.requires_grad = True

                for param in adversary.parameters():
                    param.requires_grad = False

                model = train_classifier(clf=model, optimizer_clf=optimizer_clf, adv=adversary,
                                         train_loader=dataloaders['train'], loss_criterion=loss_criterion_clf,
                                         lbda=config['mitigating']['lbda'], device=device, n_steps=100)

                for param in adversary.parameters():
                    param.requires_grad = True
            else:

                model.eval()
                adversary.eval()

                # Running parameters
                running_loss = 0.0
                running_corrects = 0
                preds_vec = np.zeros(dataset_sizes[phase])
                labels_vec = np.zeros(dataset_sizes[phase])
                probs_mat = np.zeros((dataset_sizes[phase]))
                priv = np.zeros(dataset_sizes[phase])
                cnt = 0

                # Iterate over data
                for inputs, labels, attrs in tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch}"):
                    # Send inputs and labels to the device
                    inputs = inputs.to(device)
                    labels = labels.to(device).to(torch.float)
                    attrs = attrs.to(device)

                    # Zero the gradients
                    optimizer_clf.zero_grad()

                    # Forward
                    with torch.no_grad(), autocast('cuda',enabled=(device.type == 'cuda')):
                        outputs, outputs_prev = model(inputs, training_mode=True)
                        adversary_output = adversary(outputs_prev)
                        
                        preds = outputs[:, 0] > 0.5
                        loss = loss_criterion_clf(outputs[:, 0], labels.float())
                        
                        if torch.sum(labels == 1) > 0:
                            loss = loss - config['mitigating']['lbda'] * loss_criterion_adv(
                                torch.sigmoid(adversary_output[labels == 1])[:, 0], attrs[labels == 1].float())
                        
                        probs_mat[cnt * batch_size:(cnt + 1) * batch_size] = outputs[:, 0].cpu().detach().numpy()
                        preds_vec[cnt * batch_size:(cnt + 1) * batch_size] = preds.cpu().detach().numpy()
                        labels_vec[cnt * batch_size:(cnt + 1) * batch_size] = labels.cpu().detach().numpy()
                        priv[cnt * batch_size:(cnt + 1) * batch_size] = attrs.cpu().detach().numpy()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # Increment
                    cnt = cnt + 1

                # Get the predictive performance metrics
                auroc, avg_precision, balanced_acc, f1_acc = compute_accuracy_metrics(preds_vec, labels_vec)
                bias = compute_empirical_bias(preds_vec, labels_vec, priv, bias_metric)
                logger.info(f'Bias: {bias}')
                acc = running_corrects.double() / dataset_sizes[phase]

                # Save
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = balanced_acc

                # Print
                logger.info('{} Loss: {:.4f} Acc: {:.4f} AUROC: {:.4f} Avg. Precision: {:.4f} Balanced Accuracy: {:.4f} '
                             'f1 Score: {:.4f}'.format(phase, epoch_loss, acc, auroc, avg_precision, balanced_acc, f1_acc))

                # Save the accuracy and loss
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

                # Deep copy the model
                if epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

    # Time
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Loss: {:4f}'.format(best_loss))
    logger.info('Best val Acc: {:4f}'.format(best_acc))
    logger.info('Best val Bias: {:4f}'.format(best_bias))

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, train_loss, val_acc, val_loss


def mitigating_debiasing(model_state_dict, data, config, device):
    """
    In-processing adversarial debiasing for tabular data.
    Adapted from https://github.com/abacusai/intraprocessing_debiasing
    """
    logger.info('Training Mitigating model.')
    actor = load_model(data.num_features, config.get('hyperparameters', {}))
    actor.load_state_dict(model_state_dict)
    actor.to(device)
    critic = nn.Sequential(
        nn.Linear(32, 32),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.Softmax()
    )
    critic.to(device)
    critic_optimizer = optim.Adam(critic.parameters())
    critic_loss_fn = torch.nn.BCELoss()

    actor_optimizer = optim.Adam(
        actor.parameters(), lr=config['mitigating']['lr'])
    actor_loss_fn = torch.nn.BCELoss()

    for epoch in range(config['mitigating']['epochs']):
        for param in critic.parameters():
            param.requires_grad = True
        for param in actor.parameters():
            param.requires_grad = False
        actor.eval()
        critic.train()
        for step in range(config['mitigating']['critic_steps']):
            critic_optimizer.zero_grad()
            indices = torch.randint(0, data.X_valid.size(
                0), (config['mitigating']['batch_size'],))
            cy_valid = data.y_valid_gpu[indices]
            cX_valid = data.X_valid_gpu[indices]
            cp_valid = data.p_valid_gpu[indices]
            with torch.no_grad():
                scores = actor(cX_valid)[:, 0].reshape(-1).cpu().numpy()

            res = critic(actor.trunc_forward(cX_valid))
            loss = critic_loss_fn(res[:, 0], cp_valid.type(torch.float32))
            loss.backward()
            train_loss = loss.item()
            critic_optimizer.step()
            if (epoch % 5 == 0) and (step % 100 == 0):
                logger.info(
                    f'=======> Critic Epoch: {(epoch, step)} loss: {train_loss}')

        for param in critic.parameters():
            param.requires_grad = False
        for param in actor.parameters():
            param.requires_grad = True
        actor.train()
        critic.eval()
        for step in range(config['mitigating']['actor_steps']):
            actor_optimizer.zero_grad()
            indices = torch.randint(0, data.X_valid.size(
                0), (config['mitigating']['batch_size'],))
            cy_valid = data.y_valid_gpu[indices]
            cX_valid = data.X_valid_gpu[indices]
            cp_valid = data.p_valid_gpu[indices]

            cx_predict = actor(cX_valid)
            loss_pred = actor_loss_fn(cx_predict[:, 0], cy_valid)

            cp_predict = critic(actor.trunc_forward(cX_valid))
            loss_adv = critic_loss_fn(
                cp_predict[:, 0], cp_valid.type(torch.float32))

            for param in actor.parameters():
                try:
                    lp = torch.autograd.grad(
                        loss_pred, param, retain_graph=True)[0]
                    la = torch.autograd.grad(
                        loss_adv, param, retain_graph=True)[0]
                except RuntimeError:
                    continue
                shape = la.shape
                lp = lp.flatten()
                la = la.flatten()
                lp_proj = (lp.T @ la) * la
                grad = lp - lp_proj - config['mitigating']['alpha'] * la
                grad = grad.reshape(shape)
                param.backward(grad)

            actor_optimizer.step()
            if (epoch % 5 == 0) and (step % 100 == 0):
                logger.info(f'=======> Actor Epoch: {(epoch, step)}')

        if epoch % 5 == 0:
            with torch.no_grad():
                scores = actor(data.X_valid_gpu)[
                    :, 0].reshape(-1, 1).cpu().numpy()
                _, best_mit_obj = get_best_thresh(scores, np.linspace(0, 1, 1001), data, config,
                                                  margin=config['mitigating']['margin'])
                logger.info(f'Objective: {best_mit_obj}')

    logger.info('Finding optimal threshold for Mitigating model.')
    with torch.no_grad():
        scores = actor(data.X_valid_gpu)[:, 0].reshape(-1, 1).cpu().numpy()

    best_mit_thresh, _ = get_best_thresh(scores, np.linspace(0, 1, 1001), data, config,
                                         margin=config['mitigating']['margin'])

    logger.info('Evaluating Mitigating model on best threshold.')
    with torch.no_grad():
        labels = (actor(data.X_valid_gpu)[
                  :, 0] > best_mit_thresh).reshape(-1, 1).cpu().numpy()
    results_valid = get_valid_objective(labels, data, config)
    logger.info(f'Results: {results_valid}')

    with torch.no_grad():
        labels = (actor(data.X_test_gpu)[
                  :, 0] > best_mit_thresh).reshape(-1, 1).cpu().numpy()
    results_test = get_test_objective(labels, data, config)

    return results_valid, results_test
