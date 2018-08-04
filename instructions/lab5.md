# Lab 5: Experimentation

In this lab we will introduce the IAM handwriting dataset, and give you a chance to try out different things, run experiments, and review results on W&B.

## W&B Setup

First, let's set up W&B again. We're going to do it a little differently this time. Run `wandb init`. For the team, choose fsdl, and for the project, name it `fsdl-text-recognizer-project`.

This will let us all share a project. We'll be able to see all of our runs, including network parameters and performance.


## IAM Dataset

This dataset for handwriting recognition has over 13,000 handwritten lines from 657 different writers.

Let's take a look at what it looks like in `notebooks/03-look-at-iam-lines.ipynb`.

The image width is also 952px, as in our synthetic `EmnistLines` dataset.
The maximum output length is 97 characters, however, vs. our 34 characters.

## Training

Let's train with the default params by running `tasks/train_lstm_line_predictor_on_iam.sh`.
For me, training for 8 epochs gets test set character accuracy of ~40%, and takes about 10 minutes.

Training longer will keep improving: the same settings get to 60% accuracy in 40 epochs: https://app.wandb.ai/sergeyk/fsdl-text-recognizer/runs/a6ucf77m

For the rest of the lab, let's just play around with different things and see if we can improve performance quickly.

You can see all of our training runs here: https://app.wandb.ai/fsdl/fsdl-text-recognizer-project
