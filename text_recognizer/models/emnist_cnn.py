import pathlib
from typing import Tuple

from boltons.cacheutils import cachedproperty
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

from text_recognizer.models.base import Model
from text_recognizer.datasets.emnist import EmnistDataset
from text_recognizer.networks.cnn import lenet


class EmnistCnn(Model):
    def __init__(self):
        np.random.seed(42)
        tensorflow.set_random_seed(42)

        data = EmnistDataset()
        self.mapping = data.mapping
        self.num_classes = len(self.mapping)
        self.input_shape = data.input_shape

    @cachedproperty
    def network(self):
        network = lenet(self.input_shape[0], self.input_shape[1], self.num_classes, expand_dims=True)
        network.summary()
        return network

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        # Your code below (Lab 2)
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        conf = pred_raw[ind]
        pred = self.mapping[ind]
        # Your code above (Lab 2)
        return pred, conf
