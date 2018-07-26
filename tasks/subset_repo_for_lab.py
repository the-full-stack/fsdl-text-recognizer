#!/usr/bin/evn python
import argparse
import pathlib
import shutil

import yaml


REPO_DIRNAME = pathlib.Path(__file__).parents[1].resolve()
INFO_FILENAME = REPO_DIRNAME / 'tasks' / 'lab_specific_files.yml'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lab_name', type=str, help='Name of the lab to generate subset of files for.')
    parser.add_argument('output_dirname', type=str, help='Where to output the resulting directory.')
    args = parser.parse_args()

    with open(INFO_FILENAME) as f:
        info = yaml.load(f.read())

    output_dir = pathlib.Path(args.output_dirname) / args.lab_name
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_paths = info[args.lab_name]
    for path in selected_paths:
        new_path = output_dir / path
        new_path.parents[0].mkdir(parents=True, exist_ok=True)
        shutil.copy(path, new_path)
