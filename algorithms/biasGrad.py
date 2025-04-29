"""
Bias gradient descent/ascent (GD/A) intra-processing algorithm
"""
import os.path

import numpy as np

import matplotlib.pyplot as plt

import copy

from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.amp import autocast

from sklearn.metrics import balanced_accuracy_score

import utils.data_utils
from utils.evaluation import (
    eval_model_w_data_loaders,
    find_best_threshold,
    get_valid_objective_,
    get_test_objective_,
    get_objective
)

from utils.logging_utils import get_logger

logger = get_logger("BiasGrad Model Debiasing")

def evaluate_biasgrad_model(model, dataloaders, dataset_sizes, config, device):
    """
    Perform bias gradient descent/ascent mitigation on the model.
    Returns dicts keyed by 'biasGrad' for validation and test results.
    """
    batch_size = config['biasGrad']['batch_size']
    logger.info("Starting bias gradient descent/ascent mitigation.")

    # Train a mitigated model via bias_gda_dataloaders
    mitigated_model = bias_gda_dataloaders(
        model=model,
        data_loader_train=dataloaders['val'],
        data_loader_val=dataloaders['val'],
        dataset_size_val=dataset_sizes['val'],
        opt_alg=optim.Adam,
        device=device,
        config=config,
        seed=config.get('seed', None),
        plot=False,
        display=False
    )
    logger.info("Bias mitigation completed. Evaluating model.")
    mitigated_model.to(device)
    mitigated_model.eval()

    # Collect scores
    with torch.no_grad():
        valid_scores, y_valid, p_valid = eval_model_w_data_loaders(
            model=mitigated_model,
            device=device,
            dataloader=dataloaders['val'],
            dataset_size=dataset_sizes['val'],
            batch_size=batch_size
        )
        logger.info("Validation evaluation completed.")
        test_scores, y_test, p_test = eval_model_w_data_loaders(
            model=mitigated_model,
            device=device,
            dataloader=dataloaders['test'],
            dataset_size=dataset_sizes['test'],
            batch_size=batch_size
        )
        logger.info("Test evaluation completed.")

    # Determine best threshold
    best_thresh = find_best_threshold(valid_scores, y_valid, config['acc_metric'])
    logger.info(f"Determined biasGrad best threshold: {best_thresh:.4f}")

    # Compute objectives
    valid_obj = get_valid_objective_(
        y_pred=(valid_scores > best_thresh),
        y_val=y_valid,
        p_val=p_valid,
        config=config
    )
    logger.info("Validation results (biasGrad): %s", valid_obj)

    test_obj = get_test_objective_(
        y_pred=(test_scores > best_thresh),
        y_test=y_test,
        p_test=p_test,
        config=config
    )
    logger.info(f"Test results (biasGrad): {test_obj}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    return {'biasGrad': valid_obj}, {'biasGrad': test_obj}


def spd_diff(y_pred, y_true, p):
    """Differentiable proxy function for the SPD"""
    return torch.mean(y_pred[p == 0]) - torch.mean(y_pred[p == 1])


def eod_diff(y_pred, y_true, p):
    """Differentiable proxy function for the EOD"""
    if isinstance(p, torch.Tensor):
        return torch.mean(y_pred[torch.logical_and(p == 0, y_true == 1)]) - \
               torch.mean(y_pred[torch.logical_and(p == 1, y_true == 1)])
    else:
        return torch.mean(y_pred[np.logical_and(p == 0, y_true == 1)]) - \
               torch.mean(y_pred[np.logical_and(p == 1, y_true == 1)])


def choose_best_thresh_bal_acc(data: utils.data_utils.TabularData, valid_pred_scores: np.ndarray, n_thresh=101):
    """Optimises classification threshold w.r.t. balanced accuracy"""
    threshs = np.linspace(0, 1, n_thresh)
    performances = []
    for thresh in threshs:
        perf = balanced_accuracy_score(data.y_valid, valid_pred_scores > thresh)
        performances.append(perf)
    best_thresh = threshs[np.argmax(performances)]

    return best_thresh


def choose_best_thresh_bal_acc_(y_valid: np.ndarray, valid_pred_scores: np.ndarray, n_thresh=101):
    """Optimises classification threshold w.r.t. balanced accuracy"""
    # NOTE: this function is applied directly to numpy arrays, rather than a TabularData object
    threshs = np.linspace(0, 1, n_thresh)
    performances = []
    for thresh in threshs:
        perf = balanced_accuracy_score(y_valid, valid_pred_scores > thresh)
        performances.append(perf)
    best_thresh = threshs[np.argmax(performances)]

    return best_thresh


def plot_results(step_num: np.ndarray, objective: list, bias_metric: list, pred_performance: list,
                 j_best: int, seed: int, config: dict, suffix='', display=False):
    """Plots and saves a graph with changes in the bias, performance, and constrained objective during fine-tuning"""
    fig = plt.figure()
    plt.plot(step_num, objective, label='Constrained Objective')
    plt.plot(step_num, bias_metric, label='Bias: ' + str(config['metric']))
    plt.plot(step_num, pred_performance, label='Balanced Accuracy')
    plt.vlines(x=step_num[j_best], ymin=min(bias_metric), ymax=1.0, colors='red')
    plt.xlabel('Step №')
    plt.legend()
    plt.savefig(fname=os.path.join('results/figures/biasGrad_') + str(config['experiment_name'] + '_' + str(seed) + \
                                                             suffix + '.png'), dpi=300, bbox_inches="tight")
    if display:
        plt.show()


def save_finetuning_trajectory(results: dict, seed: int, config: dict):
    """Saves traces of the bias, performance, and constrained objective during fine-tuning in a .csv file"""
    arr = np.stack((results['objective'], results['bias'], results['perf']), axis=1)
    if os.path.exists('results/logs/'):
        np.savetxt(fname=os.path.join('results/logs/') + str(config['experiment_name'] + '_' + str(seed) +
                                                                '_trajectory' + '.csv'), X=arr)
    elif os.path.exists('bin/results/logs/'):
        np.savetxt(fname=os.path.join('bin/results/logs/') + str(config['experiment_name'] + '_' + str(seed) +
                                                                 '_trajectory' + '.csv'), X=arr)
    else:
        print('WARNING: log directory is missing!')


def bias_gradient_decent(model: nn.Module, data, config: dict, device, seed: int, asc: bool = False, plot: bool = False,
                         display: bool = False):
    """Runs bias GD/A for the given model on the tabular data"""
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    logger.info('Performing bias gradient ascent/descent...\n')

    model_ = copy.deepcopy(model)

    objective_ = []
    bias_metric_ = []
    pred_performance_ = []

    best_model = None
    j_best = -1
    best_bias = 1

    model_.train()

    if config['metric'] == 'spd':
        loss_fn = spd_diff
    elif config['metric'] == 'eod':
        loss_fn = eod_diff
    else:
        raise NotImplementedError('ERROR: bias metric not supported!')

    optimiser = optim.Adam(model_.parameters(), lr=config['biasGrad']['lr'])
    logger.info(f'Using Adam optimizer with lr: {config["biasGrad"]["lr"]}')

    epoch_bar = tqdm(total=config['biasGrad']['n_epochs'], desc='Epochs')

    for i in range(config['biasGrad']['n_epochs']):
        if config['biasGrad']['val_only']:
            batch_idxs = torch.split(torch.randperm(data.X_valid.size(0)), config['biasGrad']['batch_size'])
        else:
            batch_idxs = torch.split(torch.randperm(data.X_train.size(0)), config['biasGrad']['batch_size'])
        train_loss = 0

        eval_factor = int(len(batch_idxs) / config['biasGrad']['n_evals'])
        batch_cnt = 0

        batch_bar = tqdm(batch_idxs, desc=f'Epoch {i+1} Batches', leave=False)

        for batch in batch_bar:
            if config['biasGrad']['val_only']:
                X = data.X_valid[batch, :]
                y = data.y_valid[batch]
                p = data.p_valid[batch]
            else:
                X = data.X_train[batch, :]
                y = data.y_train[batch]
                p = data.p_train[batch]

            optimiser.zero_grad()
            with autocast('cuda',enabled=(device.type == 'cuda')):
                loss = loss_fn(y_pred=model_(X)[:, 0], y_true=y, p=p)
                if asc:
                    loss = -loss
            loss.backward()
            train_loss += loss.item()
            optimiser.step()

            # Evaluate fine-tuned model on the validation set
            if batch_cnt % eval_factor == 0:
                with torch.no_grad():
                    with autocast('cuda',enabled=(device.type == 'cuda')):
                        valid_pred_scores = model_(data.X_valid)[:, 0].reshape(-1, 1).cpu().numpy()

                    # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
                    best_thresh = choose_best_thresh_bal_acc(data=data, valid_pred_scores=valid_pred_scores)

                    obj_dict = get_objective((valid_pred_scores > best_thresh) * 1., data.y_valid.numpy(), data.p_valid,
                                             config['metric'], config['objective']['sharpness'],
                                             config['objective']['epsilon'])
                    objective_.append(obj_dict['objective'])
                    bias_metric_.append(obj_dict['bias'])
                    pred_performance_.append(obj_dict['performance'])

                    # Save the model with the lowest bias that satisfies the specified accuracy constraint
                    if np.abs(obj_dict['bias']) < best_bias and obj_dict['performance'] >= config['biasGrad']['obj_lb']:
                        best_bias = np.abs(obj_dict['bias'])
                        best_model = copy.deepcopy(model_)
                        j_best = len(objective_) - 1

            batch_cnt += 1

        epoch_bar.update(1)

    # Save performance traces
    logger.info('Training complete. Saving fine-tuning trajectory...\n')
    save_finetuning_trajectory(
        results={'objective': pred_performance_ * (np.abs(bias_metric_) < config['objective']['epsilon']),
                 'bias': bias_metric_,
                 'perf': pred_performance_},
        seed=seed, config=config)

    # Plot performance traces
    if plot:
        logger.info('Plotting performance traces...')
        step_num = np.arange(1, len(objective_) + 1)
        plot_results(step_num=step_num, objective=objective_,
                     bias_metric=bias_metric_, pred_performance=pred_performance_, j_best=j_best,
                     seed=seed, config=config, display=display, suffix='')

    model_.eval()

    if best_model is None:
        logger.info('No debiased model satisfies the constraints!')
        best_model = copy.deepcopy(model)

    best_model.eval()

    return best_model


def bias_gda_dataloaders(model: nn.Module, data_loader_train: DataLoader, data_loader_val: DataLoader, dataset_size_val,
                         opt_alg, device, config: dict, seed: int, plot: bool = False,
                         display: bool = False):
    """Runs bias GD/A for the given model with the provided data loaders"""
    # NOTE: set data_loader_train to None in the call if you want to perform debiasing on the validation set only

    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    logger.info('Starting bias gradient ascent/descent with dataloaders...')

    model_ = copy.deepcopy(model)

    objective_ = []
    bias_metric_ = []
    pred_performance_ = []

    best_model = None
    j_best = -1
    best_bias = 1

    model_.train()

    if config['metric'] == 'spd':
        loss_fn = spd_diff
    elif config['metric'] == 'eod':
        loss_fn = eod_diff
    else:
        raise NotImplementedError('ERROR: bias metric not supported!')

    optimiser = opt_alg(params=model_.parameters(), lr=config['biasGrad']['lr'])

    # If training data are not provided, perform the procedure entirely on the validation data
    if data_loader_train is None:
        data_loader_train = data_loader_val
        logger.info('Training loader not provided, using validation loader for training.')

    # Evaluate the original model (in case it is already unbiased)
    logger.info('Evaluating original model...')
    with torch.no_grad():
        valid_pred_scores = np.zeros((dataset_size_val,))
        y_valid = np.zeros((dataset_size_val,))
        p_valid = np.zeros((dataset_size_val,))

        cnt = 0
        for X_, y_, p_ in tqdm(data_loader_val, desc='Validation Evaluation', leave=False):
            X_ = X_.to(device)
            y_ = y_.to(device).to(torch.float)
            p_ = p_.to(device)

            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model_(X_)

            start_idx = cnt * config['biasGrad']['batch_size']
            end_idx = (cnt + 1) * config['biasGrad']['batch_size']

            valid_pred_scores[start_idx:end_idx] = outputs[:, 0].detach().cpu().numpy()
            y_valid[start_idx:end_idx] = y_.detach().cpu().numpy()
            p_valid[start_idx:end_idx] = p_.detach().cpu().numpy()

            cnt += 1

        # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
        best_thresh = choose_best_thresh_bal_acc_(y_valid=y_valid, valid_pred_scores=valid_pred_scores)

        obj_dict = get_test_objective_(y_pred=(valid_pred_scores > best_thresh) * 1., y_test=y_valid, p_test=p_valid,
                                       config=config)
        objective_.append(obj_dict['objective'])
        bias_metric_.append(obj_dict['bias'])
        pred_performance_.append(obj_dict['performance'])

        if np.abs(obj_dict['bias']) < best_bias and obj_dict['performance'] >= config['biasGrad']['obj_lb']:
            best_bias = np.abs(obj_dict['bias'])
            best_model = copy.deepcopy(model_)
            j_best = len(objective_) - 1

        asc = obj_dict['bias'] < 0

    # Actual debiasing
    terminus_est = False
    epoch_bar = tqdm(total=config['biasGrad']['n_epochs'], desc='Epochs')
    for i in range(config['biasGrad']['n_epochs']):
        logger.info(f'Starting epoch {i+1}')
        eval_factor = int(len(data_loader_train) / config['biasGrad']['n_evals'])
        batch_cnt = 0

        # Iterate over data
        batch_bar = tqdm(data_loader_train, desc=f'Epoch {i+1} Batches', leave=False)
        for X, y, p in batch_bar:
            X = X.to(device)
            y = y.to(device).to(torch.float)
            p = p.to(device)

            optimiser.zero_grad()
            with autocast('cuda', enabled=(device.type == 'cuda')):
                loss = loss_fn(y_pred=model_(X)[:, 0], y_true=y, p=p)
                if asc:
                    loss = -loss
            loss.backward()
            optimiser.step()

            # Evaluate on the validation data periodically
            if batch_cnt % eval_factor == 0:
                with torch.no_grad():
                    valid_pred_scores = np.zeros((dataset_size_val,))
                    y_valid = np.zeros((dataset_size_val,))
                    p_valid = np.zeros((dataset_size_val,))

                    cnt = 0
                    for X_, y_, p_ in tqdm(data_loader_val, desc='Validation Eval', leave=False):
                        X_ = X_.to(device)
                        y_ = y_.to(device).to(torch.float)
                        p_ = p_.to(device)

                        with autocast('cuda', enabled=(device.type == 'cuda')):
                            outputs = model_(X_)

                        start_idx = cnt * config['biasGrad']['batch_size']
                        end_idx = (cnt + 1) * config['biasGrad']['batch_size']

                        valid_pred_scores[start_idx:end_idx] = outputs[:, 0].detach().cpu().numpy()
                        y_valid[start_idx:end_idx] = y_.detach().cpu().numpy()
                        p_valid[start_idx:end_idx] = p_.detach().cpu().numpy()

                        cnt += 1

                    # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
                    best_thresh = choose_best_thresh_bal_acc_(y_valid=y_valid, valid_pred_scores=valid_pred_scores)

                    obj_dict = get_test_objective_(y_pred=(valid_pred_scores > best_thresh) * 1., y_test=y_valid,
                                                   p_test=p_valid, config=config)
                    objective_.append(obj_dict['objective'])
                    bias_metric_.append(obj_dict['bias'])
                    pred_performance_.append(obj_dict['performance'])

                    # Save the least biased model that satisfies the specified constraint on the performance
                    if np.abs(obj_dict['bias']) < best_bias and obj_dict['performance'] >= config['biasGrad']['obj_lb']:
                        best_bias = np.abs(obj_dict['bias'])
                        best_model = copy.deepcopy(model_)
                        j_best = len(objective_) - 1

                    # Termination criterion: performance drops close to random
                    # NOTE: should be adjusted accordingly for the F1-score
                    if obj_dict['performance'] <= 0.52:
                        terminus_est = True
                        if config['acc_metric'] == 'f1_score':
                            print('\n' * 2)
                            print('WARNING: Termination criterion does not support F1-score!')
                        break

            batch_cnt += 1

        if terminus_est:
            logger.info('Early termination due to low performance.')
            break

        epoch_bar.update(1)

    epoch_bar.close()
    logger.info('Training finished. Saving performance traces...')

    # Save performance traces
    save_finetuning_trajectory(
        results={'objective': pred_performance_ * (np.abs(bias_metric_) < config['objective']['epsilon']),
                 'bias': bias_metric_,
                 'perf': pred_performance_},
        seed=seed, config=config)

    # Plot performance traces
    if plot:
        logger.info('Plotting training curves...')
        step_num = np.arange(1, len(objective_) + 1)
        plot_results(step_num=step_num, objective=objective_,
                     bias_metric=bias_metric_, pred_performance=pred_performance_, j_best=j_best,
                     seed=seed, config=config, display=display, suffix='')

    if best_model is None:
        logger.info('No debiased model satisfies the constraints. Returning initial model.')
        best_model = copy.deepcopy(model)

    model_.eval()
    best_model.eval()
    logger.info('Returning best model.')

    return best_model
