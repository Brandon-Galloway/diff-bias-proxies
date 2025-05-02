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

from sklearn.cluster import KMeans

logger = get_logger("Pruning Model Debiasing")


def evaluate_adaptive_pruning_model(model, dataloaders, dataset_sizes, config, device):
    """
    Perform structured pruning on the model and evaluate its performance.
    Returns dicts keyed by 'pruning' for validation and test results.
    """
    batch_size = config['adaptive_pruning']['batch_size']
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
    T = config['adaptive_pruning'].get('temperature', 1.5)
    valid_scores = temperature_scale(valid_scores, T)
    test_scores = temperature_scale(test_scores, T)
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


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / temperature


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
    # Coarse pass
    coarse_threshs = np.linspace(0, 1, n_thresh)
    coarse_scores = [balanced_accuracy_score(y_valid, valid_pred_scores > t) for t in coarse_threshs]
    best_thresh = coarse_threshs[np.argmax(coarse_scores)]

    # Fine-tune around best
    fine_range = np.clip(np.linspace(best_thresh - 0.05, best_thresh + 0.05, 51), 0, 1)
    fine_scores = [balanced_accuracy_score(y_valid, valid_pred_scores > t) for t in fine_range]
    best_fine_thresh = fine_range[np.argmax(fine_scores)]

    return best_fine_thresh


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
            if getattr(output, 'requires_grad', False):
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

    saliency_per_unit = []
    for i, key in enumerate(layer_keys):
        if isinstance(layers[i], nn.Linear):
            n = layers[i].out_features
        elif isinstance(layers[i], nn.Conv2d):
            act = activation[key]
            n = act.shape[1] * act.shape[2] * act.shape[3]
        else:
            raise NotImplementedError()

        saliency_per_unit.extend(coeffs[start_idx[i]:end_idx[i]].tolist())

    saliency_array = np.array(saliency_per_unit).reshape(-1, 1)
    return saliency_array, n_structs, start_idx, end_idx

def prune(model, layer_map, data_loader_train, data_loader_val, dataset_size_val, config, seed, device, arch='vgg',
          plot=False, display=False):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # prepare layers and hooks
    layers = layer_map(model)
    activation, handles = install_hooks(layers=layers)
    model.eval()

    # initial validation on unpruned model
    valid_scores = np.zeros(dataset_size_val)
    y_valid = np.zeros(dataset_size_val)
    p_valid = np.zeros(dataset_size_val)
    for i, (X, y, p) in enumerate(data_loader_val):
        X, y, p = X.to(device), y.to(device).float(), p.to(device)
        with torch.no_grad():
            out = model(X)
        start = i * config['adaptive_pruning']['batch_size']
        end = (i + 1) * config['adaptive_pruning']['batch_size']
        valid_scores[start:end] = out[:, 0].cpu().numpy()
        y_valid[start:end] = y.cpu().numpy(); p_valid[start:end] = p.cpu().numpy()
    best_t = choose_best_thresh_bal_acc_(y_valid, valid_scores)
    init_obj = get_test_objective_(y_pred=(valid_scores > best_t).astype(float), y_test=y_valid, p_test=p_valid, config=config)
    asc = init_obj['bias'] < 0

    # compute initial saliency
    logger.info('Computing initial saliency scores...')
    model.zero_grad()
    coeffs, n_structs, start_idx, end_idx = eval_saliency_dataloaders(
        model=model, layers=layers, data_loader=data_loader_train,
        activation=activation, device=device, config=config
    )

    # compute weight magnitudes aligned to coeffs
    mags = []
    for layer, count in zip(layers, n_structs):
        if isinstance(layer, torch.nn.Linear):
            w = layer.weight.detach().abs().sum(dim=1).cpu().numpy()
            mags.append(w)
        elif isinstance(layer, torch.nn.Conv2d):
            w = layer.weight.detach().abs().sum(dim=(1,2,3)).cpu().numpy()
            repeat = count // w.shape[0]
            mags.append(np.repeat(w, repeat))
        else:
            raise NotImplementedError
    weight_mags = np.concatenate(mags)

    # clustering by (saliency, weight)
    sal = coeffs.flatten()
    norm_sal = (sal - sal.min()) / (sal.ptp() + 1e-8)
    norm_wm  = (weight_mags - weight_mags.min()) / (weight_mags.ptp() + 1e-8)
    X_feat = np.stack([norm_sal, norm_wm], axis=1)
    k = config['adaptive_pruning'].get('n_clusters', 50)
    labels = KMeans(n_clusters=k).fit_predict(X_feat)

    # score clusters by signed mean saliency
    direction = 1 if asc else -1
    scores = np.array([direction * sal[labels == c].mean() for c in range(k)])
    order = np.argsort(-scores)

    # build pruning order list
    struct_order = np.concatenate([np.where(labels == c)[0] for c in order])

    # prepare masked model
    logger.info(f"Building masked model architecture: {arch}")
    if arch == 'vgg':
        model = ChestXRayVGG16Masked(model, layers, start_idx, end_idx)
    elif arch == 'resnet':
        model = ChestXRayResNet18Masked(model, layers, start_idx, end_idx)
    else:
        raise ValueError('Unsupported arch')
    model.eval()

    # iterative pruning
    pruned = []
    traj = {'objective': [], 'bias': [], 'performance': []}
    step_sz = config['adaptive_pruning']['step_size']
    for step, j in enumerate(range(0, len(struct_order), step_sz)):
        to_prune = struct_order[j:j+step_sz]
        pruned.extend(to_prune.tolist())

        # validation after prune
        valid_scores.fill(0); y_valid.fill(0); p_valid.fill(0)
        for i, (X, y, p) in enumerate(data_loader_val):
            X, y, p = X.to(device), y.to(device).float(), p.to(device)
            with torch.no_grad():
                out = model(X, pruned=np.array(pruned, dtype=int))
            start = i * config['adaptive_pruning']['batch_size']
            end = (i + 1) * config['adaptive_pruning']['batch_size']
            valid_scores[start:end] = out[:, 0].cpu().numpy()
            y_valid[start:end] = y.cpu().numpy(); p_valid[start:end] = p.cpu().numpy()
        t = choose_best_thresh_bal_acc_(y_valid, valid_scores)
        obj = get_test_objective_(y_pred=(valid_scores > t).astype(float), y_test=y_valid, p_test=p_valid, config=config)

        traj['objective'].append(obj['objective'])
        traj['bias'].append(obj['bias'])
        traj['performance'].append(obj['performance'])

        logger.info(f"Prune Step {step}: Bias={obj['bias']:.4f}, Performance={obj['performance']:.4f}, Objective={obj['objective']:.4f}")
        if step_sz * (step + 1) >= len(struct_order):
            break
        if config['adaptive_pruning'].get('dynamic', False):
            # recalc saliency on current pruned model
            model.zero_grad()
            coeffs, _, _, _ = eval_saliency_dataloaders(
                model=model, layers=layers, data_loader=data_loader_train,
                activation=activation, device=device, config=config, pruned=pruned
            )
            sal = coeffs.flatten()
            norm_sal = (sal - sal.min()) / (sal.ptp() + 1e-8)
            labels = KMeans(n_clusters=k).fit_predict(np.stack([norm_sal, norm_wm], axis=1))
            scores = np.array([direction * sal[labels == c].mean() for c in range(k)])
            order = np.argsort(-scores)
            struct_order = np.concatenate([np.where(labels == c)[0] for c in order])

    return model, lambda x: None
