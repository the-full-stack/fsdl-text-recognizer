# Text Recognizer Project

This repository is the admin view of the Full Stack Deep Learning project.

The instructions that students will see are in `instructions`.

To create the `fsdl-text-recognizer-project` directory, with files subsetted into labs and lab solutions, run `admin/tasks/subset_repo_for_labs.sh`.

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
