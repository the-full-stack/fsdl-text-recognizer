#!/bin/sh
pipenv run python training/run_experiment.py '{"dataset": "EmnistDataset", "model": "CharacterModel", "network":"mlp"}'
