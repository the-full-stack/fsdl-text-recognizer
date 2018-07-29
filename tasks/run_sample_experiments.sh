#!/bin/sh
NUM_GPUS=4
pipenv run python training/prepare_experiments.py training/experiments/sample.json | parallel -j ${NUM_GPUS}
