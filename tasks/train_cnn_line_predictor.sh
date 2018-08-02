#!/bin/sh
pipenv run python training/run_experiment.py '{"dataset": "EmnistLinesDataset", "model": "LineModel", "network": "line_cnn_sliding_window"}'
