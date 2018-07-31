#!/usr/bin/env python
import argparse
import json
import sys


def run_experiments(experiments_filename):
    with open(experiments_filename) as f:
        experiments_config = json.load(f)
    num_experiments = len(experiments_config['experiments'])
    for ind in range(num_experiments):
        experiment_config = experiments_config['experiments'][ind]
        experiment_config['experiment_group'] = experiments_config['experiment_group']
        print(f"pipenv run python training/run_experiment.py --gpu=-1 '{json.dumps(experiment_config)}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments_filename", type=str, help="Filename of JSON file of experiments to run.")
    args = parser.parse_args()
    run_experiments(args.experiments_filename)
