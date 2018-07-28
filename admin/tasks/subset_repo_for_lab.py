#!/usr/bin/env python
"""
Script to generate directories (or git branches) corresponding to subsets of the repo appropriate for different labs.

The script creates a subset of files corresponding to labs with index less than or equal than the one given,
as specified in lab_specific_files.yml

Furthermore, it also strips out text between blocks like
    # Your code below here (lab1)
    # <content>
    # Your code above here (lab1)
for labs with index greater than or equal to the one given.

"""
import argparse
import pathlib
import shutil

import yaml


REPO_DIRNAME = pathlib.Path(__file__).parents[1].resolve()
INFO_FILENAME = REPO_DIRNAME / 'tasks' / 'lab_specific_files.yml'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lab_number', type=int, help='Number of the lab to generate subset of files for.')
    parser.add_argument('output_dirname', type=str, help='Where to output the resulting directory.')
    args = parser.parse_args()

    with open(INFO_FILENAME) as f:
        info = yaml.load(f.read())

    output_dir = pathlib.Path(args.output_dirname) / f'lab{args.lab_number}'
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_paths = sum([info.get(lab_number, []) for lab_number in range(args.lab_number + 1)], [])
    print(selected_paths)
    for path in selected_paths:
        new_path = output_dir / path
        new_path.parents[0].mkdir(parents=True, exist_ok=True)
        shutil.copy(path, new_path)

    # TODO: strip out stuff between blocks
