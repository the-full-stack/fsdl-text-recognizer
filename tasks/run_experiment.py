#!/usr/bin/env python
import argparse
import json
import importlib
import os

from text_recognizer.train.gpu_manager import GPUManager
import text_recognizer.train.util as util


DEFAULT_TRAIN_ARGS = {
    'batch_size': 64,
    'epochs': 12
}


def run_experiment(filename, index, gpu_ind):
    with open(filename) as f:
        experiment_configs = json.load(f)
    experiment_config = experiment_configs[index]
    print(f'Running experiment with config {experiment_config} on GPU {gpu_ind}')

    datasets_module = importlib.import_module('text_recognizer.datasets')
    dataset_class_ = getattr(datasets_module, experiment_config['dataset'])
    dataset = dataset_class_(**experiment_config.get('dataset_args', {}))
    dataset.load_or_generate_data()
    print(dataset)

    models_module = importlib.import_module('text_recognizer.models')
    model_class_ = getattr(models_module, experiment_config['model'])
    model = model_class_(**experiment_config.get('model_args', {}))
    print(model)

    train_args = {**DEFAULT_TRAIN_ARGS, **experiment_config.get('train_args', {})}

    util.train_model(
        model=model.model,
        x_train=dataset.x_train,
        y_train=dataset.y_train,
        loss=model.loss,
        epochs=train_args['epochs'],
        batch_size=train_args['batch_size'],
        gpu_ind=args.gpu
    )
    # TODO: save model to experiment location
    score = util.evaluate_model(model.model, dataset.x_test, dataset.y_test)
    # wandb.log({'test_loss': score[0], 'test_accuracy': score[1]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Provide index of GPU to use. Providing -1 will block until there is a free one."
    )
    parser.add_argument('filename', type=str, help='Path to JSON file with experiment configs.')
    parser.add_argument('index', type=int, help='Index of experiment to run.')
    args = parser.parse_args()

    if args.gpu < 0:
        gpu_manager = GPUManager()
        args.gpu = gpu_manager.get_free_gpu()  # Blocks until one is available

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'
    run_experiment(args.filename, args.index, args.gpu)
