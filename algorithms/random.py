"""
Random perturbation intra-processing algorithm by Savani et al. (2020) [https://arxiv.org/abs/2006.08564]

Code adapted from https://github.com/abacusai/intraprocessing_debiasing
"""
import copy
from utils.logging_utils import get_logger
import math

import numpy as np

import torch
from torch.amp import autocast

from tqdm import tqdm
import time

from models.networks_tabular import load_model
from utils.evaluation import get_best_thresh, get_test_objective, get_valid_objective, eval_model_w_data_loaders, find_best_threshold, get_valid_objective_, get_test_objective_

import progressbar

logger = get_logger("Random Model Debiasing")


def evaluate_random_model(model, dataloaders, dataset_sizes, config, device):
    logger.info("Perturbing the network randomly...")
    logger.info("Configuration: num_trials=%d, stddev=%f",
                config['random']['num_trials'], config['random']['stddev'])
    start_time = time.time()

    best_score = -np.inf
    best_state_dict = None
    best_thresh = -1

    for iteration in tqdm(range(config['random']['num_trials']), desc='Random Trials', unit='trial'):
        iter_start = time.time()
        logger.info("Iteration %d/%d initializing", iteration +
                    1, config['random']['num_trials'])

        rand_model = copy.deepcopy(model).to(device)
        prev_training = rand_model.training
        rand_model.eval()

        # Perturb model's parameters
        logger.info("Iteration %d perturbing parameters", iteration + 1)
        for param in rand_model.parameters():
            param.data.mul_(torch.randn_like(param) *
                            config['random']['stddev'] + 1)

        logger.info("Iteration %d evaluating on validation set", iteration + 1)
        with torch.no_grad(), autocast(device_type=device.type):
            # Evaluate on validation set
            valid_pred_scores, y_valid, p_valid = eval_model_w_data_loaders(
                model=rand_model,
                device=device,
                dataloader=dataloaders['val'],
                dataset_size=dataset_sizes['val'],
                batch_size=config['default']['batch_size']
            )

        thresh = find_best_threshold(
            valid_pred_scores, y_valid, config['acc_metric'])
        logger.info("Iteration %d threshold=%.4f", iteration + 1, thresh)
        obj = get_valid_objective_(
            y_pred=(valid_pred_scores > thresh),
            y_val=y_valid,
            p_val=p_valid,
            config=config
        )
        logger.info("Iteration %d objective=%.4f",
                    iteration + 1, obj['objective'])

        if obj['objective'] > best_score:
            logger.info(
                "Iteration %d: new best objective %.4f (threshold %.4f)",
                iteration, obj['objective'], thresh
            )
            best_score = obj['objective']
            best_state_dict = rand_model.state_dict()
            best_thresh = thresh
            logger.info("Iteration %d new best objective=%.4f threshold=%.4f",
                        iteration + 1, best_score, best_thresh)

        if prev_training:
            rand_model.train()

        iter_elapsed = time.time() - iter_start
        logger.info("Iteration %d completed in %.2f seconds",
                    iteration + 1, iter_elapsed)

    # Evaluate the best random model
    total_elapsed = time.time() - start_time
    logger.info("Random search completed in %.2f seconds", total_elapsed)
    logger.info("Best objective after %d trials=%.4f threshold=%.4f",
                config['random']['num_trials'], best_score, best_thresh)

    logger.info(
        "Evaluating best perturbed model with threshold=%.4f", best_thresh)
    best_model = copy.deepcopy(model).to(device)
    best_model.load_state_dict(best_state_dict)
    best_model.eval()

    logger.info("Starting final evaluation")
    with torch.no_grad(), autocast(device_type=device.type):
        valid_pred_scores, y_valid, p_valid = eval_model_w_data_loaders(
            model=best_model,
            device=device,
            dataloader=dataloaders['val'],
            dataset_size=dataset_sizes['val'],
            batch_size=config['default']['batch_size']
        )
    logger.info("Validation evaluation complete")
    
    with torch.no_grad(), autocast(device_type=device.type):
        test_pred_scores, y_test, p_test = eval_model_w_data_loaders(
            model=best_model,
            device=device,
            dataloader=dataloaders['test'],
            dataset_size=dataset_sizes['test'],
            batch_size=config['default']['batch_size']
        )
    logger.info("Test evaluation complete")

    results_valid = {
        'random': get_valid_objective_(
            y_pred=(valid_pred_scores > best_thresh),
            y_val=y_valid,
            p_val=p_valid,
            config=config
        )
    }
    logger.info('Results validation: %s', results_valid['random'])

    results_test = {
        'random': get_test_objective_(
            y_pred=(test_pred_scores > best_thresh),
            y_test=y_test,
            p_test=p_test,
            config=config
        )
    }
    logger.info('Results test: %s', results_test['random'])

    del rand_model, best_model
    torch.cuda.empty_cache()
    logger.info('GPU cache cleared.')

    return results_valid, results_test


def random_debiasing(model_state_dict, data, config, device, verbose=True):
    """Runs random perturbation intra-processing, returns a perturbed network maximising the constrained objective"""
    if verbose:
        print('Perturbing the network randomly...')
        print()
    rand_model = load_model(
        data.num_features, config.get('hyperparameters', {}))
    rand_model.to(device)
    rand_result = {'objective': -math.inf,
                   'model': rand_model.state_dict(), 'thresh': -1}
    if verbose:
        bar = progressbar.ProgressBar(maxval=config['random']['num_trials'])
        bar.start()
        bar_cnt = 0
    for iteration in range(config['random']['num_trials']):
        rand_model.load_state_dict(model_state_dict)
        for param in rand_model.parameters():
            param.data = param.data * \
                (torch.randn_like(param) * config['random']['stddev'] + 1)

        rand_model.eval()
        with torch.no_grad():
            scores = rand_model(data.X_valid_gpu)[
                :, 0].reshape(-1).cpu().numpy()

        threshs = np.linspace(0, 1, 101)
        best_rand_thresh, best_obj = get_best_thresh(
            scores, threshs, data, config,  margin=config['random']['margin'])
        if best_obj > rand_result['objective']:
            rand_result = {'objective': best_obj, 'model': copy.deepcopy(rand_model.state_dict()),
                           'thresh': best_rand_thresh}
            rand_model.eval()
            with torch.no_grad():
                y_pred = (rand_model(data.X_test_gpu)[
                          :, 0] > best_rand_thresh).reshape(-1).cpu().numpy()
            best_test_result = get_test_objective(
                y_pred, data, config)['objective']

        if verbose:
            bar.update(bar_cnt)
            bar_cnt += 1

    if verbose:
        print('\n' * 2)

    rand_model.load_state_dict(rand_result['model'])
    rand_model.eval()
    with torch.no_grad():
        y_pred = (rand_model(data.X_valid_gpu)[
                  :, 0] > rand_result['thresh']).reshape(-1).cpu().numpy()
    results_valid = get_valid_objective(y_pred, data, config)

    rand_model.eval()
    with torch.no_grad():
        y_pred = (rand_model(data.X_test_gpu)[
                  :, 0] > rand_result['thresh']).reshape(-1).cpu().numpy()
    results_test = get_test_objective(y_pred, data, config)
    logger.info(f'Results: {results_test}')

    return results_valid, results_test
