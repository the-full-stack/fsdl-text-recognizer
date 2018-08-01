#!/usr/bin/env python
import argparse
import json
import importlib
import os

# Hide lines below until Lab 4
import wandb

from training.gpu_manager import GPUManager
# Hide lines above until Lab 4
from training.util import train_model


DEFAULT_TRAIN_ARGS = {
    'batch_size': 64,
    'epochs': 10
}


def run_experiment(experiment_config, save_weights, gpu_ind):
    print(f'Running experiment with config {experiment_config} on GPU {gpu_ind}')

    datasets_module = importlib.import_module('text_recognizer.datasets')
    dataset_class_ = getattr(datasets_module, experiment_config['dataset'])
    dataset = dataset_class_(**experiment_config.get('dataset_args', {}))
    dataset.load_or_generate_data()
    print(dataset)

    models_module = importlib.import_module('text_recognizer.models')
    model_class_ = getattr(models_module, experiment_config['model'])

    networks_module = importlib.import_module('text_recognizer.networks')
    network_fn_ = getattr(networks_module, experiment_config['network'])
    network_args = experiment_config.get('network_args', {})
    model = model_class_(dataset_cls=dataset_class_, network_fn=network_fn_, network_args=network_args)
    print(model)

    experiment_config['train_args'] = {**DEFAULT_TRAIN_ARGS, **experiment_config.get('train_args', {})}
    experiment_config['experiment_group'] = experiment_config.get('experiment_group', None)
    experiment_config['gpu_ind'] = gpu_ind

    # Hide lines below until Lab 4
    wandb.init()
    wandb.config.update(experiment_config)
    # Hide lines above until Lab 4

    train_model(
        model,
        dataset,
        epochs=experiment_config['train_args']['epochs'],
        batch_size=experiment_config['train_args']['batch_size']
    )
    score = model.evaluate(dataset.x_test, dataset.y_test)
    print(f'Test evaluation: {score}')

    # Hide lines below until Lab 4
    wandb.log({'test_metric': score})
    # Hide lines above until Lab 4

    if save_weights:
        model.save_weights()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Provide index of GPU to use."
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="If true, then final weights will be saved to canonical, version-controlled location"
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help="JSON of experiment to run (e.g. '{\"dataset\": \"EmnistDataset\", \"model\": \"EmnistMlp\"}'"
    )
    args = parser.parse_args()

    # Hide code below until Lab 4
    if args.gpu < 0:
        gpu_manager = GPUManager()
        args.gpu = gpu_manager.get_free_gpu()  # Blocks until one is available
    # Hide code above until Lab 4

    experiment_config = json.loads(args.experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'
    run_experiment(experiment_config, args.save, args.gpu)
