#!/usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np

from text_recognizer.datasets.emnist import EmnistDataset
from text_recognizer.models.emnist_mlp import EmnistMlp
from text_recognizer.train.util import evaluate_model, train_model


def train():
    dataset = EmnistDataset()
    model = EmnistMlp()
    train_model(model, dataset, epochs=1, batch_size=32)
    model.save_weights()
    evaluate_model(model, dataset)
    return model


if __name__ == '__main__':
    train()
