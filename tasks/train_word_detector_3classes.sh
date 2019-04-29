#!/bin/bash
PYTHONPATH=$(pwd):$PYTHONPATH pipenv run python training/run_experiment.py --gpu=1 --save '{"dataset": "IamWordsDataset", "model": "LineDetectorModel", "network": "fcn", "train_args": {"batch_size": 16, "epochs": 35}, "dataset_args": {"image_dim": 512, "num_classes": 3} }'
