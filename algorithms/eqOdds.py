import torch
from torch.amp import autocast
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from utils.evaluation import find_best_threshold, get_valid_objective_, get_test_objective_, eval_model_w_data_loaders
from utils.data_utils import to_dataframe
from utils.logging_utils import get_logger

logger = get_logger("EqOdds Model Debiasing")

def evaluate_eqod_model(model, dataloaders, dataset_sizes, config, device):
    """
    Apply Equality of Odds post-processing on model predictions.
    Returns validation and test results using the best threshold.
    """
    logger.info("Starting Equality of Odds post-processing evaluation...")

    # Prepare post-processor
    eo = EqOddsPostprocessing(
        privileged_groups=[{config['protected']: 1}],
        unprivileged_groups=[{config['protected']: 0}]
    )

    batch_size = config['default']['batch_size']

    # Obtain raw scores
    logger.info("Evaluating model on validation and test datasets.")
    with torch.no_grad(), autocast(device_type=device.type):
        valid_scores, y_valid, p_valid = eval_model_w_data_loaders(
            model=model,
            device=device,
            dataloader=dataloaders['val'],
            dataset_size=dataset_sizes['val'],
            batch_size=batch_size
        )

    with torch.no_grad(), autocast(device_type=device.type):
        test_scores, y_test, p_test = eval_model_w_data_loaders(
            model=model,
            device=device,
            dataloader=dataloaders['test'],
            dataset_size=dataset_sizes['test'],
            batch_size=batch_size
        )

    # Determine baseline threshold
    logger.info("Finding best threshold for classification.")
    best_thresh = find_best_threshold(valid_scores, y_valid, config['acc_metric'])
    logger.info(f"Best threshold determined: {best_thresh:.4f}")

    # Build AIF360 datasets
    logger.info("Converting predictions to DataFrames.")
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
    logger.info("Training Equality of Odds post-processor with validation set.")
    eo = eo.fit(val_df, val_pred_df)

    logger.info("Applying Equality of Odds to validation data.")
    val_y_pred = eo.predict(val_pred_df).labels.reshape(-1)
    results_valid = {
        'EqOdds': get_valid_objective_(
            y_pred=val_y_pred,
            y_val=y_valid,
            p_val=p_valid,
            config=config
        )
    }
    logger.info("Validation results (EqOdds): %s", results_valid['EqOdds'])

    logger.info("Applying Equality of Odds to test data.")
    test_y_pred = eo.predict(test_pred_df).labels.reshape(-1)
    results_test = {
        'EqOdds': get_test_objective_(
            y_pred=test_y_pred,
            y_test=y_test,
            p_test=p_test,
            config=config
        )
    }
    logger.info("Test results (EqOdds): %s", results_test['EqOdds'])

    # Cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    return results_valid, results_test
