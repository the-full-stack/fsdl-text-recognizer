# Text Recognizer

Project developed during lab sessions of the [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com/bootcamp).

- In this lab we will build a handwriting recognition system from scratch, and deploy it as a web service.
- We will use Keras with Tensorflow backend as the underlying framework.
- The framework will not lock us in to many design choices, and can easily be replaced with, for example, PyTorch.
- We will structure the project in a way that will scale with future development and allow us to run experiments.
- We will evaluate both convolutional and sequence methods for the task, and will see an example of how to compute loss in a more advanced way.
- Throughout development, we will submit our project code to Gradescope for autograding and leaderboard.
- We will run experiments on multiple GPUs, and store results to an online experiment management platform.
- We will set up continuous integration system for our codebase, which will check functionality of code and evaluate the model about to be deployed.
- We will package up the prediction system as a REST API, deployable as a Docker container.
- We will deploy the prediction system as a serverless function to Amazon Lambda.
- Lastly, we will set up monitoring that alerts us when the incoming data distribution changes.

## Tasks

- [ ] make Model class take a Dataset class (so that different datasets can be used with different models)

## Lab 0

- [15min] Get set up with AWS
- [5min] Get set up with Gradescope

## Lab 1 (60 min)

- [10 min] Gather handwriting data
- [5 min] Pipfile and pipenv
- [15 min] Walk through the project structure
    - talk about Dataset and uint8 vs float32 memory
    - talk about DatasetSequence and generators vs having all data in memory
        - make sure to explain that this enables a separate thread to load the data
    - talk about Keras callbacks: EarlyStopping, GPUUtilization, Tensorboard
- [15 min] They write the network in networks/mlp.py and the prediction function in character_predictor.py, and train it
- [5 min] They push code to Github and submit to gradescope for autograding

TASKS
- [ ] rename model to network
- [ ] make pre-processing logic in predict() and fit()/evaluate() should be shared
- [ ] Mostafa suggests that big datasets cannot be stored in one file. will have to shard over multiple files
- [ ] increasing GPU utiliation: can incerase batch or number of workers that preprocess data

## Lab 2 (60 min)

- [10 min] They write the CNN version of emnist_cnn.py
- [10 min] Walk through EMNIST line generation and fixed-width approach
    - Make sure to cover what the label and the loss are
- [ min] They write convnet code to use sliding window (they write sliding window part)
- [ min] They write convnet code to be all conv and observe whether it's faster or not

- [ ] make sure to specify that padding has to be 'same'
- [ ] make a slide explaining the all-conv idea

## Lab 3 (60 min)

- [10 min] They write the LSTM version

- [ ] be able to answer what the CTC token is doing
- [ ] be able to answer how loss is backprop'd through: to the first repeated char or for all of them?
- [ ] write output shapes for all lines in the network

## Lab 4 (30 min)

- Weights & Biases and script to distribute jobs over multiple GPUs
- [ ] talk about distributing training of a single model (in addition to multiple experiments)
mple
## Lab 5 (60 min)

- [10 min] Introduce IAM dataset
- [10 min] Introduce image preprocessing via https://keras.io/preprocessing/image/#image-preprocessing
- [40 min] More-or-less free lab to try to get the highest character accuracy, via searching over model space and augmenting data generator
    - They can set their best model to be the official line_predictor model and submit to Gradescope to leaderboard it

## Lab 6 (30 min)

- [10 min] Adding CI via CircleCI
- [10 min] Running a Flask web app locally
- [10 min] Dockerizing the flask web app

## Lab 7 (60 min)

- Deploying to lambda
- Seeing it work on your phone via cursive.ai
- Add monitoring dashboard and alarms

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
    api/                        # Code for serving predictions as a REST API.
        app.py                      # Flask web server that serves predictions.
        Dockerfile                  # Specificies Docker image that runs the web server.
        serverless.yml              # Specifies AWS Lambda deployment of the REST API.

    data/                       # Data for training. Not under version control.
        raw/                        # The raw data source. Perhaps from an external source, perhaps from your DBs. Contents of this should be re-creatable via scripts.
            emnist-matlab.zip
        processed/                  # Data in a format that can be used by our Dataset classses.
            byclass.npz

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
