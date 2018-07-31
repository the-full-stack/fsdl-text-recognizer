import pathlib
from typing import Tuple

from boltons.cacheutils import cachedproperty
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

from text_recognizer.models.base import Model
from text_recognizer.datasets.emnist import EmnistDataset
from text_recognizer.networks.mlp import mlp


class EmnistMlp(Model):
    def __init__(self, dataset, network_fn, layer_size: int=128, dropout_amount: float=0.2):
        np.random.seed(42)
        tensorflow.set_random_seed(42)

        data = EmnistDataset()
        self.mapping = data.mapping
        self.num_classes = len(self.mapping)
        self.input_shape = data.input_shape

        self.layer_size = layer_size
        self.dropout_amount = dropout_amount

    @cachedproperty
    def network(self):
        network = mlp(self.num_classes, self.input_shape, self.layer_size, self.dropout_amount)
        network.summary()
        return network

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        # Your code below (Lab 1)
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        conf = pred_raw[ind]
        pred = self.mapping[ind]
        # Your code above (Lab 1)
        return pred, conf
