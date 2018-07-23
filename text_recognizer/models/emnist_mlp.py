import pathlib
from typing import Tuple

import numpy as np
import tensorflow
from tensorflow.python.keras.models import load_model, Model, Sequential
from tensorflow.python.keras.layers import Activation, Dense, Dropout

from text_recognizer.datasets.emnist import Emnist


DIRNAME = pathlib.Path(__file__).parents[0].resolve()
MODEL_WEIGHTS_FILENAME = DIRNAME / 'emnist_mlp_weights.h5'


def _create_mlp(num_classes: int,
                input_size: int,
                layer_size: int=128,
                dropout_amount: float=0.2) -> Model:
    model = Sequential()
    model.add(Dense(layer_size, activation='relu', input_shape=(input_size,)))
    model.add(Dropout(dropout_amount))
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dropout(dropout_amount))
    model.add(Dense(num_classes, activation='softmax'))
    return model


class EmnistMlp:
    def __init__(self, layer_size: int=128, dropout_amount: float=0.2):
        data = EmnistDataset()
        self.mapping = data.mapping
        self.num_classes = len(self.mapping)
        self.input_size = data.input_size

        np.random.seed(42)
        tensorflow.set_random_seed(42)

        self.model = _create_mlp(self.num_classes, self.input_size, layer_size, dropout_amount)
        self.model.summary()

    def load_weights(self):
        self.model.load_weights(str(MODEL_WEIGHTS_FILENAME))

    def save_weights(self):
        self.model.save_weights(str(MODEL_WEIGHTS_FILENAME))

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        pred_raw = self.model.predict(image.reshape(1, -1), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        conf = pred_raw[ind]
        pred = self.mapping[ind]
        return pred, conf

    loss = 'categorical_crossentropy'
