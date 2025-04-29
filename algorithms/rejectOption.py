import torch
from torch.amp import autocast
from aif360.algorithms.postprocessing import RejectOptionClassification
from utils.evaluation import find_best_threshold, get_valid_objective_, get_test_objective_, eval_model_w_data_loaders
from utils.data_utils import to_dataframe
from utils.logging_utils import get_logger

logger = get_logger("ROC Model Debiasing")

def evaluate_roc_model(model, dataloaders, dataset_sizes, config, device):
    """
    Apply Reject Option Classification post-processing on model predictions.
    Returns validation and test results using the best threshold.
    """
    metric_map = {
        'spd': 'Statistical parity difference',
        'aod': 'Average odds difference',
        'eod': 'Equal opportunity difference'
    }

    logger.info("Starting ROC post-processing evaluation...")

    # Prepare post-processor
    roc = RejectOptionClassification(
        unprivileged_groups=[{config['protected']: 1}],
        privileged_groups=[{config['protected']: 0}],
        low_class_thresh=0.01,
        high_class_thresh=0.99,
        num_class_thresh=100,
        num_ROC_margin=50,
        metric_name=metric_map[config['metric']],
        metric_ub=config['objective']['epsilon'],
        metric_lb=-config['objective']['epsilon']
    )

    # Raw scores
    logger.info("Evaluating model on validation and test datasets.")
    with torch.no_grad(), autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
        valid_scores, y_valid, p_valid = eval_model_w_data_loaders(
            model=model,
            device=device,
            dataloader=dataloaders['val'],
            dataset_size=dataset_sizes['val'],
            batch_size=config['default']['batch_size']
        )
        test_scores, y_test, p_test = eval_model_w_data_loaders(
            model=model,
            device=device,
            dataloader=dataloaders['test'],
            dataset_size=dataset_sizes['test'],
            batch_size=config['default']['batch_size']
        )

    # Determine baseline threshold
    logger.info("Finding best threshold for classification.")
    best_thresh = find_best_threshold(valid_scores, y_valid, config['acc_metric'])
    logger.info(f"Best threshold determined: {best_thresh:.4f}")

    # Build AIF360 datasets
    logger.info("Converting validation and test predictions to DataFrames.")
    val_df = to_dataframe(y_true=y_valid, y_pred=valid_scores, y_prot=p_valid, prot_name=config['protected'])
    val_pred_df = to_dataframe(
        y_true=(valid_scores > best_thresh).astype(float),
        y_pred=valid_scores,
        y_prot=p_valid,
        prot_name=config['protected']
    )
    test_pred_df = to_dataframe(
        y_true=(test_scores > best_thresh).astype(float),
        y_pred=test_scores,
        y_prot=p_test,
        prot_name=config['protected']
    )

    # Fit and predict
    logger.info("Training ROC post-processor with validation set.")
    roc = roc.fit(val_df, val_df)

    logger.info("Applying ROC post-processing to validation data.")
    val_y_pred = roc.predict(val_pred_df).labels.reshape(-1)
    results_valid = {
        'ROC': get_valid_objective_(
            y_pred=val_y_pred,
            y_val=y_valid,
            p_val=p_valid,
            config=config
        )
    }
    logger.info(f"Validation results after ROC: {results_valid['ROC']}")

    logger.info("Applying ROC post-processing to test data.")
    test_y_pred = roc.predict(test_pred_df).labels.reshape(-1)
    results_test = {
        'ROC': get_test_objective_(
            y_pred=test_y_pred,
            y_test=y_test,
            p_test=p_test,
            config=config
        )
    }
    logger.info(f"Test results after ROC: {results_test['ROC']}")

    # Cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    return results_valid, results_test
