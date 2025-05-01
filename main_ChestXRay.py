"""
Runs MIMIC-CXR experiments
"""
import argparse
import json
from pathlib import Path

import numpy as np

import torch

import yaml
import os

from algorithms.default import evaluate_default_model
from algorithms.random import evaluate_random_model
from algorithms.rejectOption import evaluate_roc_model
from algorithms.eqOdds import evaluate_eqod_model
from algorithms.adversarial import (evaluate_adversarial_model)
from algorithms.mitigating import evaluate_mitigating_model
from algorithms.pruning import evaluate_pruning_model
from algorithms.biasGrad import evaluate_biasgrad_model
from algorithms.adaptivePruning import evaluate_adaptive_pruning_model


from datasets.chestxray_dataset import get_ChestXRay_mimic_dataloaders

from models.networks_ChestXRay import (
    train_ChestXRay_model, ChestXRayResNet18, ChestXRayVGG16)

from torch import optim
from torch import nn
from torch.optim import lr_scheduler

from utils.logging_utils import get_logger

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

logger = get_logger("Debiasing")

def init_checkpoint(exp_name, seed, config, results_dir='results/logs'):
    valid_path = Path(results_dir) / f'{exp_name}_valid_output_{seed}.json'
    test_path  = Path(results_dir) / f'{exp_name}_test_output_{seed}.json'

    if valid_path.exists():
        with open(valid_path) as f:
            results_valid = json.load(f)
    else:
        results_valid = {}

    if test_path.exists():
        with open(test_path) as f:
            results_test = json.load(f)
    else:
        results_test = {}

    def save_checkpoint():
        results_valid['config'] = config
        with open(valid_path, 'w') as f:
            json.dump(results_valid, f)
        results_test['config'] = config
        with open(test_path, 'w') as f:
            json.dump(results_test, f)

    return results_valid, results_test, save_checkpoint


def main(config):
    seeds = [np.random.randint(0, high=10000)]
    if 'seed' in config:
        seeds = config['seed']

    # NOTE: replace with relevant directories
    if config['dataset'] == 'chestxray_mimic':
        ROOT_DIR = Path(config['root_dir'])
    else:
        NotImplementedError('This chest X-ray dataset not supported!')

    # Set up dataloader cache in advance
    batch_sizes = {config['default']['batch_size']}
    for key in ['adversarial','mitigating','biasGrad','pruning', 'adaptive_pruning']:
        if key in config['models']:
            batch_sizes.add(config[key]['batch_size'])

    

    for seed in seeds:
        logger.info(f'Running the experiment for seed: {seed}.')
        torch.manual_seed(seed)
        np.random.seed(seed)

        results_valid, results_test, save_checkpoint = init_checkpoint(
            config['experiment_name'], seed, config
        )

        loader_cache = {bs: get_ChestXRay_mimic_dataloaders(
            device=device,
            root_dir=ROOT_DIR,
            prot_attr=config['protected'],
            priv_class=config['priv_class'],
            unpriv_class=config['unpriv_class'],
            train_prot_ratio=config['prot_ratio'],
            class_names=[config['disease'], 'No Finding'],
            batch_size=bs,
            num_workers=config['num_workers'],
            seed=seed
        ) for bs in batch_sizes}

        # Setup directories to save models and results
        Path('models').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)
        Path('results/figures').mkdir(exist_ok=True)
        Path('results/logs').mkdir(exist_ok=True)

        # Get a pretrained model
        if config['default']['arch'] == 'resnet':
            model = ChestXRayResNet18(pretrained=config['default']['pretrained'])
        elif config['default']['arch'] == 'vgg':
            model = ChestXRayVGG16(pretrained=config['default']['pretrained'])
        else:
            ValueError('Network architecture not supported!')
        
        # Load the initial (default) loaders
        dataloaders, dataset_sizes = loader_cache[config['default']['batch_size']]

        model = model.to(device)

        model_path = os.path.join(
            'results', 'models', config['modelpath'] + str('_') + str(seed) + '.pt')
        if Path(model_path).is_file():
            logger.info(f'Loading Model from {model_path}.')
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            logger.info(f'Training model from scratch.')

            optimizer = optim.AdamW(
                model.parameters(), lr=1e-4, weight_decay=1e-8)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            model, train_acc, train_loss, val_acc, val_loss = train_ChestXRay_model(
                dataloaders=dataloaders, dataset_sizes=dataset_sizes, model=model, criterion=nn.BCELoss(),
                optimizer=optimizer, scheduler=scheduler, device=device, class_names=[
                    config['disease'], 'No Finding'],
                bias_metric=config['metric'], batch_size=config['default']['batch_size'],
                num_epochs=config['default']['n_epochs'])
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)

        # Preliminaries
        logger.info('Setting up preliminaries.')
        model.eval()

        # Evaluate the default model
        if 'default' in config['models']:
            if 'default' in results_valid and 'default' in results_test:
                logger.info('Skipping Default Evaluation (already done).')
            else:
                logger.info('Beginning Default Evaluation...')
                results_valid['default'], results_test['default'] = evaluate_default_model(
                    model=model,
                    dataloaders=dataloaders,
                    dataset_sizes=dataset_sizes,
                    config=config,
                    device=device
                )
                save_checkpoint()

        # Evaluate random perturbation intra-processing
        if 'random' in config['models']:
            if 'random' in results_valid and 'random' in results_test:
                logger.info('Skipping random Evaluation (already done).')
            else:
                logger.info('Beginning Random Evaluation...')
                results_valid['random'], results_test['random'] = evaluate_random_model(
                    model=model,
                    dataloaders=dataloaders,
                    dataset_sizes=dataset_sizes,
                    config=config,
                    device=device
                )
                save_checkpoint()

        # Evaluate the ROC post-processing
        if 'ROC' in config['models']:
            if 'ROC' in results_valid and 'ROC' in results_test:
                logger.info('Skipping ROC Evaluation (already done).')
            else:
                logger.info('Beginning ROC Evaluation...')
                results_valid['ROC'], results_test['ROC'] = evaluate_roc_model(
                    model=model,
                    dataloaders=dataloaders,
                    dataset_sizes=dataset_sizes,
                    config=config,
                    device=device
                )
                save_checkpoint()

        # Evaluate the equality of odds post-processing
        if 'EqOdds' in config['models']:
            if 'EqOdds' in results_valid and 'EqOdds' in results_test:
                logger.info('Skipping EqOdds Evaluation (already done).')
            else:
                logger.info('Beginning EqOdds Evaluation...')
                results_valid['EqOdds'], results_test['EqOdds'] = evaluate_eqod_model(
                    model=model,
                    dataloaders=dataloaders,
                    dataset_sizes=dataset_sizes,
                    config=config,
                    device=device
                )
                save_checkpoint()

        # Evaluate adversarial intra-processing
        if 'adversarial' in config['models']:
            if 'adversarial' in results_valid and 'adversarial' in results_test:
                logger.info('Skipping adversarial Evaluation (already done).')
            else:
                logger.info('Beginning adversarial Evaluation...')
                dataloaders_adv, _ = loader_cache[config['adversarial']['batch_size']]
                results_valid['adversarial'], results_test['adversarial'] = evaluate_adversarial_model(
                    model=model,
                    dataloaders=dataloaders_adv,
                    dataset_sizes=dataset_sizes,
                    config=config,
                    device=device
                )
                save_checkpoint()

        # Evaluate adversarial in-processing
        if 'mitigating' in config['models']:
            if 'mitigating' in results_valid and 'mitigating' in results_test:
                logger.info('Skipping mitigating Evaluation (already done).')
            else:
                logger.info('Beginning mitigating Evaluation...')
                dataloaders_mit, _ = loader_cache[config['mitigating']['batch_size']]
                results_valid['mitigating'], results_test['mitigating'] = evaluate_mitigating_model(
                    model=model,
                    dataloaders=dataloaders_mit,
                    dataset_sizes=dataset_sizes,
                    config=config,
                    device=device
                )
                save_checkpoint()

        # Evaluate bias gradient descent/ascent
        if 'biasGrad' in config['models']:
            if 'biasGrad' in results_valid and 'biasGrad' in results_test:
                logger.info('Skipping biasGrad Evaluation (already done).')
            else:
                logger.info('Beginning biasGrad Evaluation...')
                dataloaders_bg, _ = loader_cache[config['biasGrad']['batch_size']]
                results_valid['biasGrad'], results_test['biasGrad'] = evaluate_biasgrad_model(
                    model=model,
                    dataloaders=dataloaders_bg,
                    dataset_sizes=dataset_sizes,
                    config=config,
                    device=device
                )
                save_checkpoint()

        # NOTE: needs to be run the last, makes adjustments to the model object directly (to use less memory)
        # Evaluate pruning
        if 'pruning' in config['models']:
            if 'pruning' in results_valid and 'pruning' in results_test:
                logger.info('Skipping pruning Evaluation (already done).')
            else:
                logger.info('Beginning pruning Evaluation...')
                dataloaders_pr, _ = loader_cache[config['pruning']['batch_size']]
                with torch.no_grad():
                    state_backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                
                try:
                    results_valid['pruning'], results_test['pruning'] = evaluate_pruning_model(
                        model=model,
                        dataloaders=dataloaders_pr,
                        dataset_sizes=dataset_sizes,
                        config=config,
                        device=device
                    )
                    pruned_path = os.path.join('results', 'models',f"{config['modelpath']}_pruned_{seed}.pt")
                    Path(pruned_path).parent.mkdir(parents=True, exist_ok=True)
                    with torch.no_grad():
                        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, pruned_path)
                    logger.info(f'Saved pruned model to {pruned_path}')
                finally:
                    model.cpu()
                    model.load_state_dict(state_backup)
                    model.to(device)
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        logger.info("GPU cache cleared.")
                save_checkpoint()

        if 'adaptive_pruning' in config['models']:
            if 'adaptive_pruning' in results_valid and 'adaptive_pruning' in results_test:
                logger.info('Skipping adaptive_pruning Evaluation (already done).')
            else:
                logger.info('Beginning adaptive_pruning Evaluation...')
                dataloaders_apr, _ = loader_cache[config['adaptive_pruning']['batch_size']]
                with torch.no_grad():
                    state_backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                try:
                    results_valid['adaptive_pruning'], results_test['adaptive_pruning'] = evaluate_adaptive_pruning_model(
                        model,
                        dataloaders_apr,
                        dataset_sizes,
                        config,
                        device,
                    )
                    pruned_path = os.path.join(
                        'results', 'models',
                        f"{config['modelpath']}adaptive_pruning{seed}.pt"
                    )
                    Path(pruned_path).parent.mkdir(parents=True, exist_ok=True)
                    with torch.no_grad():
                        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, pruned_path)
                    save_checkpoint()
                finally:
                    model.cpu()
                    model.load_state_dict(state_backup)
                    model.to(device)
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        logger.info('GPU cache cleared.')

        logger.info(f'Validation Results: {results_valid}')
        logger.info(f'Test Results: {results_test}')

        del model, dataloaders
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to configuration yaml file.')
    args = parser.parse_args()
    with open(args.config, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    # Load the preprocessing config and extract the root_dir
    with open('./configs/preprocessing.yml', 'r') as fh:
        preprocessing_cfg = yaml.load(fh, Loader=yaml.FullLoader)

    # Inject preprocessing root_dir
    mimic_cfg = preprocessing_cfg.get('MIMIC', {})
    mimic_root = mimic_cfg.get('root_dir')
    output_subdir = mimic_cfg.get('output_subdir')
    if not mimic_root or not output_subdir:
        raise ValueError(
            "Missing 'root_dir' or 'output_subdir' in MIMIC section of preprocessing config.")
    config['root_dir'] = str(Path(mimic_root) / output_subdir)

    main(config)
