# Text Recognizer

Project developed during lab sessions of the [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com/bootcamp).

## Quick Start

```
# Get development environment set up
pipenv install --dev

# Train EMNIST MLP with default settings
pipenv run tasks/train_emnist_mlp.py

# Run EMNIST MLP experiments
pipenv run tasks/run_emnist_mlp_experiments.py

# Update the EMNIST MLP model to deploy by taking the best model from the experiments
pipenv run tasks/update_model_with_best_experiment.py --name='emnist_mlp'

# Test EMNIST MLP model
pipenv run pytest --pyargs text_recognizer

# Evaluate EMNIST MLP model
pipenv run tasks/evaluate_emnist_mlp_model.py

# Run the API server
pipenv run python web/server.py

# Deploy the server to AWS
pipenv run tasks/deploy_web_server_to_aws.py
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
            test_emnist_mlp_predict.py  # Lightweight test to ensure that the trained emnist_mlp correctly classifies a few data points.

        web/                        # Web server for serving predictions.
            server.py

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

## Tasks

### July 18

Today, I want to do all parts of training, testing, ci, and web server deploying for just EMNIST MLP.
I should be able to, from scratch: train EMNIST MLP, test on a few important examples, evaluate on test set, and deploy as a flask web api.

## Commands I want to be able to run

```
# Train MLP on EMNIST with default parameters
bin/train --name='emnist_mlp' --model='emnist_mlp' --dataset='emnist'

train/emnist_mlp.py

# Run training on MLP


# Deploy MLP as a web service
bin/deploy --name='emnist_mlp'
```
