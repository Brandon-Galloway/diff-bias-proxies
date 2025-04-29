import numpy as np
import torch
from tqdm import tqdm
from utils.evaluation import find_best_threshold, get_valid_objective_, get_test_objective_

from utils.logging_utils import get_logger

logger = get_logger("Default Model Debiasing")

def evaluate_default_model(model, dataloaders, dataset_sizes, config, device):
    logger.info('Finding best threshold for default model to minimize objective function')
    logger.info('Starting evaluation of the default model.')

    val_scores = np.zeros(dataset_sizes['val'])
    val_labels = np.zeros(dataset_sizes['val'])
    val_prot   = np.zeros(dataset_sizes['val'])

    test_scores = np.zeros(dataset_sizes['test'])
    test_labels = np.zeros(dataset_sizes['test'])
    test_prot   = np.zeros(dataset_sizes['test'])

    batch_size = config['default']['batch_size']

    with torch.no_grad():
        logger.info('Processing validation set (%d samples)', dataset_sizes['val'])
        for idx, (inputs, labels, attrs) in enumerate(tqdm(dataloaders['val'], desc='Validation', unit='batch')):
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            attrs  = attrs.to(device)

            outputs = model(inputs)

            start = idx * batch_size
            end   = start + inputs.size(0)

            val_scores[start:end] = outputs[:, 0].cpu().numpy()
            val_labels[start:end] = labels.cpu().numpy()
            val_prot[start:end]   = attrs.cpu().numpy()

        logger.info('Validation loop completed.')

        logger.info('Processing test set (%d samples)', dataset_sizes['test'])
        for idx, (inputs, labels, attrs) in enumerate(tqdm(dataloaders['test'], desc='Test', unit='batch')):
            # send inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            attrs  = attrs.to(device)

            outputs = model(inputs)

            start = idx * batch_size
            end   = start + inputs.size(0)

            test_scores[start:end] = outputs[:, 0].cpu().numpy()
            test_labels[start:end] = labels.cpu().numpy()
            test_prot[start:end]   = attrs.cpu().numpy()

        logger.info('Test loop completed.')

    logger.info('Determining best threshold for %s.', config['acc_metric'])
    best_thresh = find_best_threshold(val_scores, val_labels, config['acc_metric'])
    logger.info('Best threshold found: %.4f', best_thresh)


    logger.info('Evaluating default model with best threshold.')
    valid_res = get_valid_objective_(y_pred=(val_scores > best_thresh),
                                     y_val=val_labels,
                                     p_val=val_prot,
                                     config=config)
    test_res  = get_test_objective_(y_pred=(test_scores > best_thresh),
                                    y_test=test_labels,
                                    p_test=test_prot,
                                    config=config)
    logger.info(f'Results: {test_res}')

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info('GPU cache cleared.')

    return valid_res, test_res
