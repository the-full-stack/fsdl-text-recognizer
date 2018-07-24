#!/usr/bin/env python
import json
import sys


def run_experiments(filename):
    with open(filename) as f:
        experiments_config = json.load(f)
    num_experiments = len(experiments_config['experiments'])
    for ind in range(num_experiments):
        print(f'pipenv run python tasks/run_experiment.py --gpu=-1 {filename} {ind}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: ./run_experiments.py <experiments.json>")
        sys.exit(1)
    run_experiments(sys.argv[1])
