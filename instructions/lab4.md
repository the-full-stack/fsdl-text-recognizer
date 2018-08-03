# Lab 4

In this lab, we will get familiar with Weights & Biases, and start using an experiment-running framework that will make it easy to distribute work onto multiple GPUs.

The first thing to do is run `wandb init`, where you can set your user id and create a new project.

Now, if you look at `training/run_experiment.py` file, you'll notice that we are going to be syncing to W&B.
