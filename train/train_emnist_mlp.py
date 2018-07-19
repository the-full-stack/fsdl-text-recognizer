"""
Script for training MLP on EMNIST.
"""
from text_recognizer.datasets.emnist import Emnist
from text_recognizer.models.emnist_mlp import EmnistMlp
from util import evaluate_model, train_model


def train_emnist_mlp():
    data = Emnist()
    model = EmnistMlp()

    train_model(
        model=model.model,
        x_train=data.x_train,
        y_train=data.y_train,
        loss=model.loss,
        epochs=3,
        batch_size=256
    )
    model.save_weights()

    evaluate_model(model.model, data.x_test, data.y_test)

    return model


if __name__ == '__main__':
    train_emnist_mlp()
