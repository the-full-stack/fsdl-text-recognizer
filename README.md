# Text Recognizer

Project developed during lab sessions of the [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com/bootcamp).

## Tasks / Ideas

- [?] Refactor emnist_predictor to character_predictor

- One strategy for how to progress through the labs is creating git branches via script:
    - the subset_repo_for_lab script can generate branches instead of directories
    - [ ] Strip out stuff between "your code here" lines in generating the lab subsets

- [ ] make list of suggestions for what to try in the experiment lab

- Looking into dataset augmentation:
    - Don't think it makes sense to use tf.data.Dataset. Can point them to it for homework, though.
    - Do need a solution for separating the input/output transformation code from the augmentation code.
        - What probably makes the most sense is passing InputOutputFormatter and DataAugmenter as arguments to Dataset
        - [x] refactor CtcDatasetSequence to simply pass a formatting function to DatasetSequence

## Lab 0

- [15min] Get set up with AWS
- [5min] Get set up with Gradescope

## Lab 1

- [10 min] Gather handwriting data
- [10 min] Walk through the project structure
- [ min] They write the network in networks/mlp.py
- [ min] They write the prediction function in emnist_predictor.py
- [ min] They submit their thing to gradescope for autograding

## Lab 2

- [10 min] They write the CNN version of char_model.py
- [10 min] Walk through EMNIST line generation
- [ min] They write fixed-width and TimeDistributed code
- [ min] They write convnet code to use sliding window (they write sliding window part)
- [ min] They write convnet code to be all conv and observe whether it's faster or not

## Lab 3

- [10 min] They write the LSTM version

## Lab 4

## Lab 5

- [10 min] Introduce IAM dataset
- [40 min] More-or-less free lab to try to get the highest character accuracy, via searching over model space and augmenting data generator

## Quick Start

```
export PYTHONPATH=.  # May want to put this in your .bashrc

# Get development environment set up
# First, make sure you are using whatever Python you intend to use (conda or system).
# Then you can install packages via pipenv.
pipenv --python=`which python`
pipenv install --dev

# Train EMNIST MLP with default settings
pipenv run train/train_emnist_mlp.py

# Run experiments from a config file, in parallel, on available GPUs
pipenv run tasks/prepare_experiments.py experiments/experiments.json.sample | parallel -j4

# Update the EMNIST MLP model to deploy by taking the best model from the experiments
pipenv run train/update_model_with_best_experiment.py --name='emnist_mlp'

# Test EMNIST MLP model
pipenv run pytest text_recognizer

# Evaluate EMNIST MLP model
pipenv run train/evaluate_emnist_mlp_model.py

# Run the API server
pipenv run python web/server.py

# Build the API server docker image
docker build -t text_recognizer -f text_recognizer/Dockerfile .

# Run the API server via docker
docker run -p 8000:8000 text_recognizer

# Make a sample request to the running API server
# TODO

# Deploy the server to AWS
pipenv run tasks/deploy_web_server_to_aws.py
```

## W&B

```
pipenv run wandb login

pipenv run wandb init
# set up new project
```

## Project Structure

```
text_recognizer/
    data/                       # Data for training. Not under version control.
        raw/                        # The raw data source. Perhaps from an external source, perhaps from your DBs. Contents of this should be re-creatable via scripts.
            emnist-matlab.zip
        processed/                  # Data in a format that can be used by our Dataset classses.
            emnist-byclass.npz

    experiments/                # Not under code version control.
        emnist_mlp/                 # Name of the experiment
            models/
            logs/

    notebooks/                  # For snapshots of initial exploration, before solidfying code as proper Python files.
        00-download-emnist.ipynb    # Naming pattern is <order>-<initials>-<description>.ipynb
        01-train-emnist-mlp.ipynb

    text_recognizer/            # Package that can be deployed as a self-contained prediction system.
        __init__.py

        datasets/                   # Code for loading datasets
            __init__.py
            emnist.py

        models/                     # Code for instantiating models, including data preprocessing and loss functions
            __init__.py
            emnist_mlp.py               # Code
            emnist_mlp.h5               # Learned weights
            emnist_mlp.config           # Experimental config that led to the learned weights

        predict/
            __init__.py
            emnist_mlp.py

        test/                       # Code that tests functionality of the other code.
            support/                    # Support files for the tests
                emnist/
                    a.png
                    3.png
            test_emnist_predict.py  # Lightweight test to ensure that the trained emnist_mlp correctly classifies a few data points.

        web/                        # Web server for serving predictions.
            api.py

    tasks/
        train_emnist_mlp.py
        run_emnist_mlp_experiments.py
        update_model_with_best_experiment.py
        evaluate_emnist_mlp_model.py
        tasks/deploy_web_server_to_aws.py

    train/                       # Code for running training experiments and selecting the best model.
        run_experiment.py           # Script for running a training experiment.
        gpu_manager.py              # Support script for distributing work onto multiple GPUs.
        select_best_model.py        # Script for selecting the best model out of multiple experimental instances.

    Pipfile
    Pipfile.lock
    README.md
    setup.py
```

## Explanation

### Pipenv

Pipenv is necessary for being exact about the dependencies.
TODO: explain that want to stay up to date with packages, but only update them intentionally, not randomly. Explain sync vs install.

```
# Workhorse command when adding another dependency
pipenv install --dev --keep-outdated

# Periodically, update all versions
pipenv install --dev

# For deployment, no need to install dev packages
pipenv install
```
