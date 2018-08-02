#!/usr/bin/env python
"""
Script to generate directories (or git branches) corresponding to subsets of the repo appropriate for different labs.

The script creates a subset of files corresponding to labs with index less than or equal than the one given,
as specified in lab_specific_files.yml

Furthermore, it also strips out text between blocks like
    ##### Your code below (Lab1)
    # <content>
    ##### Your code above (Lab1)
for labs with index greater than or equal to the one given.

It also strips text between blocks like
    ###### Hide lines below until Lab 2
    # <content>
    ##### Hide lines above until Lab 2
for labs with index greater than the one given.

NOTE that the stripping is only performed on .py files.
"""
import argparse
import os
import glob
import pathlib
import re
import shutil

import yaml

MAX_LAB_NUMBER = 10
REPO_DIRNAME = pathlib.Path(__file__).parents[2].resolve()
INFO_FILENAME = REPO_DIRNAME / 'admin' / 'tasks' / 'lab_specific_files.yml'


def _filter_your_code_blocks(lines, lab_number):
    """
    Strip out stuff between "Your code here" blocks.
    """
    lab_numbers_to_strip = f"[{'|'.join(str(num) for num in range(lab_number, MAX_LAB_NUMBER))}]"
    beginning_comment = f'##### Your code below \(Lab {lab_numbers_to_strip}\)'
    ending_comment = f'##### Your code above \(Lab {lab_numbers_to_strip}\)'
    filtered_lines = []
    filtering = False
    for line in lines:
        if not filtering:
            filtered_lines.append(line)
        if re.search(beginning_comment, line):
            filtering = True
            filtered_lines.append('')
        if re.search(ending_comment, line):
            filtered_lines.append(line)
            filtering = False
    return filtered_lines


def _filter_hidden_blocks(lines, lab_number):
    lab_numbers_to_hide = f"[{'|'.join(str(num) for num in range(lab_number + 1, MAX_LAB_NUMBER))}]"
    beginning_comment = f'##### Hide lines below until Lab {lab_numbers_to_hide}'
    ending_comment = f'##### Hide lines above until Lab {lab_numbers_to_hide}'
    filtered_lines = []
    filtering = False
    for line in lines:
        if re.search(beginning_comment, line):
            filtering = True
        if re.search(ending_comment, line):
            filtering = False
            continue
        if not filtering:
            filtered_lines.append(line)
    return filtered_lines


def _replace_data_dirname(lines):
    filtered_lines = []
    for line in lines:
        if line == "        return pathlib.Path(__file__).parents[2].resolve() / 'data'":
            line = "        return pathlib.Path(__file__).parents[3].resolve() / 'data'"
        filtered_lines.append(line)
    return filtered_lines


def _copy_files_for_lab(info, lab_number, lab_output_dir):
    selected_paths = sum([info.get(number, []) for number in range(lab_number + 1)], [])
    new_paths = []
    for path in selected_paths:
        new_path = lab_output_dir / path
        new_path.parents[0].mkdir(parents=True, exist_ok=True)
        shutil.copy(path, new_path)
        new_paths.append(new_path)
    return new_paths


def _process_new_files(new_paths, lab_number, filter_your_code=True, filter_hidden=True, replace_data_dirname=True):
    for path in new_paths:
        if path.suffix != '.py':
            continue

        with open(path) as f:
            lines = f.read().split('\n')

        if filter_your_code:
            lines = _filter_your_code_blocks(lines, lab_number)
        if filter_hidden:
            lines = _filter_hidden_blocks(lines, lab_number)
        if replace_data_dirname:
            lines = _replace_data_dirname(lines)

        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


def subset_repo(info):
    """See module docstring."""
    output_dir = pathlib.Path(args.output_dirname)
    if output_dir.exists():
        for d in glob.glob(f'{str(output_dir)}/lab*'):
            shutil.rmtree(d)
        if os.path.exists(output_dir / 'data'):
            shutil.rmtree(output_dir / 'data')
        if os.path.exists(output_dir / 'Pipfile'):
            os.remove(output_dir / 'Pipfile')
        if os.path.exists(output_dir / 'Pipfile.lock'):
            os.remove(output_dir / 'Pipfile.lock')

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_DIRNAME / 'data', output_dir / 'data')
    shutil.copy('Pipfile', output_dir)
    shutil.copy('Pipfile.lock', output_dir)
    shutil.copy('.gitignore', output_dir)

    # To be filled-out
    for lab_number in info.keys():
        lab_output_dir = output_dir / f'lab{lab_number}'
        lab_output_dir.mkdir(parents=True)
        new_paths = _copy_files_for_lab(info, lab_number, lab_output_dir)
        _process_new_files(new_paths, lab_number)

    # Solutions
    for lab_number in info.keys():
        lab_output_dir = output_dir / f'lab{lab_number}_sln'
        lab_output_dir.mkdir(parents=True)
        new_paths = _copy_files_for_lab(info, lab_number, lab_output_dir)
        _process_new_files(new_paths, lab_number, filter_your_code=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirname', default='_labs', help='Where to output the lab subset directories.')
    args = parser.parse_args()

    with open(INFO_FILENAME) as f:
        info = yaml.load(f.read())

    subset_repo(info)
