"""
Script for training MLP on EMNIST.
"""
import pathlib

from text_recognizer.datasets.emnist import EMNIST
from text_recognizer.models.mlp import create_mlp_model
from text_recognizer.train.util import train_model


GPU_IND = None
REPO_DIRNAME = pathlib.Path(__file__).parents[1].resolve()
MODELS_DIRNAME = REPO_DIRNAME / 'text_recognizer' / 'models' / 'emnist_mlp'


def train_emnist_mlp():
    data = EMNIST()
    num_classes = data.y_train.shape[1]
    input_size = data.x_train.shape[1]

    model = create_mlp_model(num_classes, input_size)

    history = train_model(
        model=model,
        x_train=data.x_train,
        y_train=data.y_train,
        loss='categorical_crossentropy',
        epochs=1,
        batch_size=256,
        gpu_ind=GPU_IND
    )

    score = model.evaluate(data.x_test, data.y_test, verbose=1)
    print('Test loss/accuracy:', score[0], score[1])

    filename = f'model.h5'
    model.save(MODELS_DIRNAME / filename)

    return model


if __name__ == '__main__':
    train_emnist_mlp()
