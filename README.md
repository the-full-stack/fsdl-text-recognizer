# Text Recognizer

Project developed during lab sessions of the [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com/bootcamp).

## Log
- [x] Make a Flask JSON API (input is image URL)
- [x] Modify Emnist dataset class to store mapping separately, in text_recognizer/datasets
- [x] Get web server to run in a Docker container
- [x] Deploy web server to AWS ECS via Docker or to AWS Lambda via nice script

- July 20 2330: Decided to go with Lambda instead. ECS scales too slowly. Lambda will be cooler.
    - But first going to try Elastic Beanstalk real quick
    - For lambda, do the building and the zipping in a Docker container
        - serverless makes it really easy: https://serverless.com/framework/docs/providers/aws/events/apigateway/#simple-http-endpoint
            - https://medium.com/tooso/serving-tensorflow-predictions-with-python-and-aws-lambda-facb4ab87ddd

- July 21 2300: spent whole day but figured out deployment to lambda
- [x] Make cursive.ai a Flask web site where user can upload image and then direct it to an API URL passed a query string
- July 21 2300: decided that this should also be a serverless app
- Realized it's nice and easy to deploy a Flask app via https://github.com/logandk/serverless-wsgi, which makes for nice dev environment
- [x] Re-deploy prediction API using WSGI plugin, so that it's less of a delta from the Flask web app to deploying on Lambda
- [x] add synthetic line dataset (import code from all-in-one notebook)

- july 22 2330: reading about tf.estimator and tf.dataset stuff. tf has really developed, this is nice: https://www.tensorflow.org/tutorials/

- july 23 0200: ported most of the synthetic line code, should be going to sleep now
- [x] add self-caching to speed up loads
- [x] train all-conv network on line synthetic dataset
    - will probably have to convert data to tensorflow sooner rather than later -- but it's not something that *has* to be done
    - [x] move convert_preds_to_string to line_cnn model
july 23 1230: finished up basic CNN and caching EmnistLinesDataset
- [x] overlap parameter is a single number that sets max overlap, with uniform distribution up to it
july 23 1720: figured out FC vs 1x1 conv, need to run it on the farm though
- [x] make EmnistDataset give you images instead of vectors (and use HDF5)
July 23 2030: currently running sliding window experiments with CNNs
- [x] test different cnn archs (sliding window, FCs on top, timeditributed on top instead of bottom) on the farm

July 24 0030:
    - thinking that worst case if we don't get individual aws credits, can set up IAM users, make everyone a machine with limited access just to that machine, and tell them to name their Lambda functions with their emails, such that they don't conflict with each other
    - also thinking that I won't be able to get to detection
July 24 0120 Thinking about this order now:
    - Plumbing: handwriting samples, guided tour through the project, and submit trained EmnistMlp to Gradescope
    - Convnets: guided tour of creating EmnistLines synthetic dataset, then show the FixedWidth way, then increase dataset overlap and give them a chance to figure out tf.extract_image_patches, then a chance to figure out all-conv solution, then submit any trained model to Gradescope
    - Sequences: guided tour of IAM test set, then of the RNN solution, give them a chance to figure out CTC loss, then time to play around with dataset augmentation to get leaderboard on Gradescope
    - Infrastructure: guided tour of the experiment-running framework, GPU manager, and Weights & Biases
    - Free lab: a full hour to play around with what we have so far and try to get highest score
    - Deployment 1: write tests, set up Docker, CI, and Flask-based web server, just running on local
    - Deployment 2: Guided tour of serverless and lambda-based deploy. They should all have their own endpoint running. Everyone tests out cursive.ai, then we switch up the image processing and add monitoring.
July 24 0230
    Got a nice experimental framework coded up and working on the farm!
    Next up: sync results to W&B. Then implement LineRNN
July 24 1345
- syncing results to W&B nicely, now going to implement LineRNN and then work on the convnet lecture

July 24 2100
- Able to successfully train LSTM with CTC loss now, thanks to Saurabh
NEXT UP:
- [ ] have the Model be responsible for training itself (because CTC needs special outputs in the dataset)
- [ ] implement predict functionality for CTC model (as in notebook on the farm)
- [ ] add character-accuracy metric to the CTC model for displaying in training and evaluation
- [ ] add IAM dataset

## Tasks

Detection
- [ ] write out instructions for composing a test set of paragraph images
- [ ] add paragraph synthetic dataset: 2h
- [ ] train detection network to detect lines in the paragraph images: 3h
- [ ] IB make web app for annotating paragraph images with line locations

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
            test_emnist_mlp_predict.py  # Lightweight test to ensure that the trained emnist_mlp correctly classifies a few data points.

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
