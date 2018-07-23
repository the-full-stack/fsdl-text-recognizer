import numpy as np

from text_recognizer.datasets.emnist_lines import EmnistLinesDataset
from text_recognizer.models.line_cnn import LineCnn
from text_recognizer.train.util import evaluate_model, train_model


def train():
    dataset = EmnistLinesDataset(num_train=100, num_test=10)
    model = LineCnn()

    train_model(
        model=model.model,
        x_train=dataset.x_train,
        y_train=dataset.y_train,
        loss=model.loss,
        epochs=3,
        batch_size=128
    )
    model.save_weights()

    evaluate_model(model.model, dataset.x_test, dataset.y_test)

    return model


if __name__ == '__main__':
    train()
