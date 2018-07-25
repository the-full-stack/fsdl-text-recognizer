import pathlib
from typing import Tuple

import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

from text_recognizer.models.base import Model
from text_recognizer.datasets.emnist import EmnistDataset
from text_recognizer.networks.mlp import mlp


class EmnistMlp(Model):
    def __init__(self, layer_size: int=128, dropout_amount: float=0.2):
        np.random.seed(42)
        tensorflow.set_random_seed(42)

        data = EmnistDataset()
        self.mapping = data.mapping
        self.num_classes = len(self.mapping)
        self.input_shape = data.input_shape
        self.model = mlp(self.num_classes, self.input_shape, layer_size, dropout_amount)
        self.model.summary()

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        pred_raw = self.model.predict(np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        conf = pred_raw[ind]
        pred = self.mapping[ind]
        return pred, conf

    loss = 'categorical_crossentropy'
