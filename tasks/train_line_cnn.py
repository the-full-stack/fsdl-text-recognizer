#!/usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np

from text_recognizer.datasets.emnist_lines import EmnistLinesDataset
from text_recognizer.models.line_cnn import LineCnnFixedWidth, LineCnnSlidingWindow
from text_recognizer.train.util import evaluate_model, train_model


def train():
    dataset = EmnistLinesDataset()
    dataset.load_or_generate_data()
    model = LineCnnFixedWidth()
    # model = LineCnnSlidingWindow()
    train_model(model, dataset, epochs=2, batch_size=32)
    model.save_weights()
    evaluate_model(model, dataset)
    return model


if __name__ == '__main__':
    train()
