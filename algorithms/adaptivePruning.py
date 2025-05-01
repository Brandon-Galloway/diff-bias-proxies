"""
Pruning intra-processing algorithm
"""
import os.path

import numpy as np

import copy

import torch
from torch import nn
from torch.amp import autocast

from sklearn.metrics import balanced_accuracy_score

import utils.data_utils
from models.networks_ChestXRay import (ChestXRayVGG16Masked, ChestXRayResNet18Masked, get_layers_to_prune_ResNet18, get_layers_to_prune_VGG16)

from utils.evaluation import (get_objective, get_test_objective_, eval_model_w_data_loaders, find_best_threshold, get_valid_objective_)

from utils.plotting import plot_pruning_results
from utils.logging_utils import get_logger

from tqdm import tqdm

from collections import OrderedDict

from typing import Dict, Callable

logger = get_logger("Pruning Model Debiasing")


def evaluate_adaptive_pruning_model(model, dataloaders, dataset_sizes, config, device):
    """
    Perform structured pruning on the model and evaluate its performance.
    Returns dicts keyed by 'pruning' for validation and test results.
    """
    batch_size = config['pruning']['batch_size']
    # Select layer map based on architecture
    arch = config['default']['arch']
    if arch == 'resnet':
        layer_map = get_layers_to_prune_ResNet18
    elif arch == 'vgg':
        layer_map = get_layers_to_prune_VGG16
    else:
        raise ValueError('Network architecture not supported!')

    logger.info("Starting structured pruning using architecture '%s'.", arch)
    # Perform pruning
    model_pruned, pruned_fn = prune(
        model,
        layer_map,
        dataloaders['val'],
        dataloaders['val'],
        dataset_sizes['val'],
        config,
        seed=config.get('seed', None),
        device=device,
        plot=False,
        display=False
    )

    logger.info("Pruning complete. Moving model to device and evaluating.")
    model_pruned.to(device)
    model_pruned.eval()

    # Evaluate pruned model
    with torch.no_grad():
        valid_scores, y_valid, p_valid = eval_model_w_data_loaders(
            model=model_pruned,
            device=device,
            dataloader=dataloaders['val'],
            dataset_size=dataset_sizes['val'],
            batch_size=batch_size,
            forward_args=[pruned_fn]
        )
        logger.info("Validation evaluation completed.")
        test_scores, y_test, p_test = eval_model_w_data_loaders(
            model=model_pruned,
            device=device,
            dataloader=dataloaders['test'],
            dataset_size=dataset_sizes['test'],
            batch_size=batch_size,
            forward_args=[pruned_fn]
        )
        logger.info("Test evaluation completed.")

    logger.info("Finding best threshold for pruned model.")
    best_thresh = find_best_threshold(valid_scores, y_valid, config['acc_metric'])
    logger.info("Best threshold for pruning: %.4f", best_thresh)

    valid_obj = get_valid_objective_(
        y_pred=(valid_scores > best_thresh),
        y_val=y_valid,
        p_val=p_valid,
        config=config
    )
    logger.info("Validation results (pruning): %s", valid_obj)

    test_obj = get_test_objective_(
        y_pred=(test_scores > best_thresh),
        y_test=y_test,
        p_test=p_test,
        config=config
    )
    logger.info("Test results (pruning): %s", test_obj)

    # Cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    return {'pruning': valid_obj}, {'pruning': test_obj}





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


def save_pruning_trajectory(results: dict, seed: int, config: dict):
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


def install_hooks(layers, names=None, get_n_units=False):
    activation = {}
    handles_or_units = {}

    def get_activation(name):
        def hook(_, __, output):
            output.retain_grad()
            activation[name] = output
        return hook

    for i, layer in enumerate(layers):
        name = names[i] if names else f'l{i}'
        handles_or_units[name] = (
            layer.out_features if get_n_units else layer.register_forward_hook(get_activation(name))
        )

    return activation, handles_or_units


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    """Removes all forward hooks from the given model"""
    for child in model.children():
        child._forward_hooks.clear()
        remove_all_forward_hooks(child)


def eval_saliency(model: nn.Module, data: utils.data_utils.TabularData, idx: np.ndarray,
                  activation: dict, config: dict, total_n_units: int, val_only=False):
    model.eval()
    model.zero_grad()

    X = data.X_valid if val_only else data.X_train
    preds = model(X[idx])[:, 0]

    coeffs = np.zeros(total_n_units)

    if config['metric'] == 'spd':
        p = data.p_valid if val_only else data.p_train
        bias_measure = preds[p[idx] == 0].mean() - preds[p[idx] == 1].mean()
    elif config['metric'] == 'eod':
        p = data.p_valid if val_only else data.p_train
        y = data.y_valid if val_only else data.y_train
        mask = y[idx] == 1
        bias_measure = preds[np.logical_and(p[idx] == 0, mask)].mean() - preds[np.logical_and(p[idx] == 1, mask)].mean()
    else:
        raise NotImplementedError('Bias metric not supported!')

    bias_measure.backward()

    offset = 0
    grads_fc0 = activation['fc0'].grad.sum(0).cpu().numpy()
    coeffs[offset:offset + model.fc0.out_features] = grads_fc0
    offset += model.fc0.out_features

    for i, fc in enumerate(model.fcs):
        grads = activation[f'fc{i + 1}'].grad.sum(0).cpu().numpy()
        coeffs[offset:offset + fc.out_features] = grads
        offset += fc.out_features

    return coeffs


def eval_saliency_dataloaders(model, layers, data_loader, activation, device, config, pruned=None):
    logger.info('Evaluating gradient-based bias influence (saliency scores)...')
    
    model.eval()

    dummy_input = torch.zeros((1, 3, 224, 224), device=device)
    _ = model(dummy_input)

    total_n_structs, n_structs, start_idx, end_idx = 0, [], [], []

    for i, layer in enumerate(layers):
        start_idx.append(total_n_structs)
        key = f'l{i}'
        if isinstance(layer, nn.Linear):
            n = layer.out_features
        elif isinstance(layer, nn.Conv2d):
            act = activation[key]
            n = act.shape[1] * act.shape[2] * act.shape[3]
        else:
            raise NotImplementedError('Layer type not supported!')
        n_structs.append(n)
        total_n_structs += n
        end_idx.append(total_n_structs)
        logger.info(f'Layer {i}: {layer.__class__.__name__} with {n} prunable units.')

    coeffs = np.zeros(total_n_structs)
    logger.info(f'Total prunable structures: {total_n_structs}')

    if pruned is not None:
        pruned_array = np.array(pruned)

    layer_keys = [f'l{i}' for i in range(len(layers))]

    for inputs, labels, attrs in tqdm(data_loader, desc='Evaluating Saliency', leave=False):
        model.zero_grad()

        X = inputs.to(device)
        y = labels.to(device).float()
        p = attrs.to(device)

        with autocast(device_type=device.type):
            outputs = model(X) if pruned is None else model(X, pruned=pruned_array)

        preds = outputs[:, 0]

        if config['metric'] == 'spd':
            bias_measure = preds[p == 0].mean() - preds[p == 1].mean()
        elif config['metric'] == 'eod':
            mask = y == 1
            bias_measure = preds[torch.logical_and(p == 0, mask)].mean() - preds[torch.logical_and(p == 1, mask)].mean()
        else:
            raise NotImplementedError('Bias metric not supported!')

        if not torch.isnan(bias_measure):
            bias_measure.backward()
            for i, key in enumerate(layer_keys):
                grad = activation[key].grad
                if isinstance(layers[i], nn.Linear):
                    sal = grad.mean(0)
                elif isinstance(layers[i], nn.Conv2d):
                    sal = grad.mean(0).flatten()
                else:
                    raise NotImplementedError('Layer type not supported!')
                coeffs[start_idx[i]:end_idx[i]] += sal.detach().cpu().numpy()

    return coeffs, n_structs, start_idx, end_idx



def prune_fc_units(model: nn.Module, to_prune: np.ndarray, n_units: dict, prune_first=False):
    """Prunes specified units in the fully connected layers by adjusting weight matrices"""
    if len(to_prune) == 0:
        return model
    cnt = 0
    if prune_first:
        n_units_i = n_units['fc0']
        # Find units to prune in this layer
        pruned_units_i = to_prune[np.logical_and(cnt <= to_prune, to_prune < cnt + n_units_i)] - cnt
        # Set incoming weights to 0
        model.fc0.weight[pruned_units_i, :] = 0
        model.fc0.bias[pruned_units_i] = 0
        cnt += n_units_i
    for (i, fc) in enumerate(model.fcs):
        n_units_i = n_units['fc' + str(i + 1)]
        # Find units to prune in this layer
        pruned_units_i = to_prune[np.logical_and(cnt <= to_prune, to_prune < cnt + n_units_i)] - cnt
        # Set incoming weights to 0
        fc.weight[pruned_units_i, :] = 0
        fc.bias[pruned_units_i] = 0
        cnt += n_units_i
    return model


def prune_fc(model, data, config, seed, plot=False, display=False):
    """Intra-processing debiasing procedure for pruning fully connected neural networks"""
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logger.info('Starting fully connected network pruning...')

    model_pruned = copy.deepcopy(model)

    # Determine the order in which structures are pruned, similar to bias GD/A
    # Predict on validation data
    with torch.enable_grad():
        # Predict on validation set
        valid_pred_scores = model_pruned(data.X_valid)[:, 0].reshape(-1, 1).detach().numpy()
    # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
    best_thresh = choose_best_thresh_bal_acc(data=data, valid_pred_scores=valid_pred_scores)
    # Evaluate all metrics using the best threshold
    obj_dict = get_objective((valid_pred_scores > best_thresh) * 1., data.y_valid.numpy(), data.p_valid,
                             config['metric'], config['objective']['sharpness'],
                             config['objective']['epsilon'])
    asc = obj_dict['bias'] < 0

    # Create hooks to get layer activations from the model
    activation, n_units = install_hooks_fc(model=model_pruned)

    total_n_units = sum(list(n_units.values()))

    if not config['pruning']['val_only']:
        idx = np.arange(0, data.X_train.size(0))
    else:
        idx = np.arange(0, data.X_valid.size(0))

    # Evaluate unit gradient-based bias influence
    coeffs = eval_saliency(model=model_pruned, data=data, idx=idx, activation=activation, config=config,
                           total_n_units=total_n_units, val_only=config['pruning']['val_only'])

    # Sort the units according to their influence and the sign of the initial bias
    if asc:
        unit_inds = np.argsort(coeffs)
    else:
        unit_inds = np.argsort(-coeffs)

    model_pruned.eval()

    objective = []
    bias_metric = []
    pred_performance = []
    n_pruned = []
    pruned_inds = []
    pruned = []
    model_pruned_ = copy.deepcopy(model_pruned)
    start_ind = 0

    j_best = -1
    best_bias = 1

    prune_bar = tqdm(total=int((len(unit_inds) + 1) / config['pruning']['step_size']), desc='Pruning Steps')
    with torch.no_grad():
        # Prune units step-by-step measuring performance changes for every sparsity level
        for j in range(0, len(unit_inds) + 1, config['pruning']['step_size']):
            # Recompute influence dynamically for pruned networks
            logger.info(f'Pruning step {j // config["pruning"]["step_size"]}: Starting...')
            
            if config['pruning']['dynamic'] and j > 1:
                logger.info(f'Pruning step {j // config["pruning"]["step_size"]}: Recomputing saliency scores...')
                with torch.enable_grad():
                    coeffs = eval_saliency(model=model_pruned_, data=data, idx=idx, activation=activation,
                                           config=config, total_n_units=total_n_units,
                                           val_only=config['pruning']['val_only'])
                    if asc:
                        unit_inds = np.argsort(coeffs)
                    else:
                        unit_inds = np.argsort(-coeffs)
                logger.info(f'Pruning step {j // config["pruning"]["step_size"]}: Saliency scores recomputed.')

            # NOTE: evaluate unpruned network as well (in case it is not biased)
            if j > 0:
                # Prune top salient units
                to_prune = unit_inds[start_ind:(start_ind + config['pruning']['step_size'])]
            else:
                to_prune = []

            if not config['pruning']['dynamic']:
                start_ind += config['pruning']['step_size']

            logger.info(f'Pruning step {j // config["pruning"]["step_size"]}: Pruning {len(to_prune)} units...')

            for unit in tqdm(to_prune, desc=f'Pruning Units (Step {j // config["pruning"]["step_size"]})', leave=False):
                pruned.append(unit)
            
            pruned_inds.append(copy.deepcopy(pruned))
            # Prune the network
            model_pruned_ = prune_fc_units(model=model_pruned_, to_prune=to_prune, n_units=n_units, prune_first=True)

            # Predict on the validation set
            logger.info(f'Pruning step {j // config["pruning"]["step_size"]}: Running validation forward pass...')
            with torch.enable_grad():
                valid_pred_scores = model_pruned_(data.X_valid)[:, 0].reshape(-1, 1).detach().numpy()

            # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
            logger.info(f'Pruning step {j // config["pruning"]["step_size"]}: Searching for best threshold...')
            best_thresh = choose_best_thresh_bal_acc(data=data, valid_pred_scores=valid_pred_scores)

            # Evaluate all metrics using the best threshold
            logger.info(f'Pruning step {j // config["pruning"]["step_size"]}: Evaluating performance metrics...')
            obj_dict = get_objective((valid_pred_scores > best_thresh) * 1., data.y_valid.numpy(), data.p_valid,
                                     config['metric'], config['objective']['sharpness'],
                                     config['objective']['epsilon'])

            objective.append(obj_dict['objective'])
            bias_metric.append(obj_dict['bias'])
            pred_performance.append(obj_dict['performance'])
            n_pruned.append(j)

            # Save the least biased model that satisfies the specified constraint on the performance
            logger.info(f'Pruning step {j // config["pruning"]["step_size"]}: '
                    f'Bias={obj_dict["bias"]:.4f}, Performance={obj_dict["performance"]:.4f}, Objective={obj_dict["objective"]:.4f}')
            if np.abs(obj_dict['bias']) < best_bias and obj_dict['performance'] >= config['pruning']['obj_lb']:
                best_bias = np.abs(obj_dict['bias'])
                j_best = len(objective) - 1

            # Stop pruning if accuracy drops below 52%
            if config['pruning']['stop_early'] and obj_dict['performance'] < 0.52:
                logger.info('Early stopping: performance dropped too low.')
                logger.warning('WARNING: Early stopping does not support F1-score!')
                prune_bar.close()
                break

            prune_bar.update(1)

        prune_bar.close()

        if j_best == -1:
            logger.info('WARNING: No debiased model satisfies the constraints!')
            j_best = 0

        # Plot performance traces
        if plot:
            logger.info('Plotting pruning results...')
            plot_pruning_results(n_pruned=n_pruned, total_n_units=total_n_units, objective=objective,
                                 bias_metric=bias_metric, pred_performance=pred_performance, j_best=j_best,
                                 seed=seed, config=config, display=display)

        # Save performance traces
        logger.info('Saving pruning trajectory...')
        save_pruning_trajectory(
            results={'objective': pred_performance * (np.abs(bias_metric) < config['objective']['epsilon']),
                     'bias': bias_metric,
                     'perf': pred_performance},
            seed=seed, config=config)

        # List of units pruned in the optimal model
        to_prune = np.array(pruned_inds[j_best])

        # Construct the best model
        logger.info('Building final pruned model.')
        with torch.no_grad():
            model_pruned = copy.deepcopy(model)
            model_pruned.eval()

            # Prune the final model
            model_pruned = prune_fc_units(model=model_pruned, to_prune=to_prune, n_units=n_units, prune_first=True)
    logger.info('Pruning complete. Returning final model.')
    return model_pruned


def prune(model, layer_map, data_loader_train, data_loader_val, dataset_size_val, config, seed, device, arch='vgg',
          plot=False, display=False):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    layers = layer_map(model)
    activation, handles = install_hooks(layers=layers)
    model.eval()

    logger.info('Starting network pruning...')
    logger.info('Evaluating original model before pruning...')

    valid_pred_scores, y_valid, p_valid = np.zeros(dataset_size_val), np.zeros(dataset_size_val), np.zeros(dataset_size_val)

    for i, (X, y, p) in enumerate(tqdm(data_loader_val, desc='Initial Validation', leave=False)):
        X, y, p = X.to(device), y.to(device).float(), p.to(device)
        with torch.enable_grad():
            outputs = model(X)
        start = i * config['pruning']['batch_size']
        end = (i + 1) * config['pruning']['batch_size']
        valid_pred_scores[start:end] = outputs[:, 0].detach().cpu().numpy()
        y_valid[start:end] = y.detach().cpu().numpy()
        p_valid[start:end] = p.detach().cpu().numpy()

    best_thresh = choose_best_thresh_bal_acc_(y_valid=y_valid, valid_pred_scores=valid_pred_scores)
    obj_dict = get_test_objective_(
        y_pred=(valid_pred_scores > best_thresh).astype(float),
        y_test=y_valid,
        p_test=p_valid,
        config=config
    )

    objective, bias_metric, pred_performance = [obj_dict['objective']], [obj_dict['bias']], [obj_dict['performance']]
    n_pruned, pruned_inds, pruned = [[0]], [[]], []
    j_best, best_bias, to_prune_best = -1, 1.0, None
    asc = obj_dict['bias'] < 0

    logger.info('Computing initial saliency scores...')
    model.zero_grad()
    coeffs, n_structs, start_idx, end_idx = eval_saliency_dataloaders(
        model=model, layers=layers, data_loader=data_loader_train,
        activation=activation, device=device, config=config
    )
    struct_inds = np.argsort(coeffs if asc else -coeffs)

    logger.info(f'Building masked model architecture: {arch}')
    if arch == 'vgg':
        model = ChestXRayVGG16Masked(model, layers, start_idx, end_idx)
    elif arch == 'resnet':
        model = ChestXRayResNet18Masked(model, layers, start_idx, end_idx)
    else:
        raise ValueError('Network architecture not supported!')

    step_num = 0
    prune_bar = tqdm(total=(len(struct_inds) + 1) // config['pruning']['step_size'], desc='Pruning Steps')

    for j in range(0, len(struct_inds), config['pruning']['step_size']):
        if config['pruning']['dynamic'] and j > 0:
            logger.info('Recomputing saliency scores...')
            model.zero_grad()
            coeffs, _, _, _ = eval_saliency_dataloaders(
                model=model, layers=layers, data_loader=data_loader_train,
                activation=activation, device=device, config=config, pruned=pruned
            )
            struct_inds = np.argsort(coeffs if asc else -coeffs)

        to_prune = struct_inds[len(pruned):len(pruned) + config['pruning']['step_size']]
        pruned.extend(to_prune)
        pruned_inds.append(pruned.copy())

        valid_pred_scores.fill(0)
        y_valid.fill(0)
        p_valid.fill(0)

        for i, (X, y, p) in enumerate(tqdm(data_loader_val, desc=f'Validation after {len(pruned)} Prunes', leave=False)):
            model.zero_grad()
            X, y, p = X.to(device), y.to(device).float(), p.to(device)
            with torch.enable_grad():
                outputs = model(X, pruned=np.array(pruned))
            start = i * config['pruning']['batch_size']
            end = (i + 1) * config['pruning']['batch_size']
            valid_pred_scores[start:end] = outputs[:, 0].detach().cpu().numpy()
            y_valid[start:end] = y.detach().cpu().numpy()
            p_valid[start:end] = p.detach().cpu().numpy()

        best_thresh = choose_best_thresh_bal_acc_(y_valid=y_valid, valid_pred_scores=valid_pred_scores)
        obj_dict = get_test_objective_(
            y_pred=(valid_pred_scores > best_thresh).astype(float),
            y_test=y_valid,
            p_test=p_valid,
            config=config
        )

        objective.append(obj_dict['objective'])
        bias_metric.append(obj_dict['bias'])
        pred_performance.append(obj_dict['performance'])
        n_pruned.append(j)

        if abs(obj_dict['bias']) < best_bias and obj_dict['performance'] >= config['pruning']['obj_lb']:
            best_bias, j_best, to_prune_best = abs(obj_dict['bias']), len(objective) - 1, pruned.copy()

        logger.info(f"Prune Step {j // config['pruning']['step_size']}: Bias={obj_dict['bias']:.4f}, "
                    f"Performance={obj_dict['performance']:.4f}, Objective={obj_dict['objective']:.4f}")

        if config['pruning']['stop_early']:
            if obj_dict['performance'] <= 0.55:
                logger.info('Early stopping: performance too low.')
                break
            if step_num >= config['pruning']['max_steps']:
                logger.info('Early stopping: reached max steps.')
                break

        step_num += 1
        prune_bar.update(1)

    prune_bar.close()

    if plot:
        logger.info('Plotting pruning results...')
        plot_pruning_results(
            n_pruned=n_pruned, total_n_units=len(coeffs), objective=objective,
            bias_metric=bias_metric, pred_performance=pred_performance, j_best=j_best,
            seed=seed, config=config, display=display
        )

    logger.info('Saving pruning performance trajectory...')
    save_pruning_trajectory({
        'objective': pred_performance * (np.abs(bias_metric) < config['objective']['epsilon']),
        'bias': bias_metric,
        'perf': pred_performance
    }, seed=seed, config=config)

    logger.info('Cleaning up hooks...')
    remove_all_forward_hooks(model)
    for h in handles.values():
        h.remove()
    
    if to_prune_best is None:
        logger.warning('No valid pruned model met the performance constraint. Returning original model with no pruning.')
        to_prune_best = np.array([], dtype=int)

    return model, np.array(to_prune_best)
