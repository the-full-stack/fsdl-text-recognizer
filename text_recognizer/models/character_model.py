import pathlib
from typing import Callable, Dict, Tuple

from boltons.cacheutils import cachedproperty
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

from text_recognizer.models.base import Model
from text_recognizer.datasets.emnist import EmnistDataset
from text_recognizer.networks.mlp import mlp
from text_recognizer.networks.cnn import lenet


class CharacterModel(Model):
    def __init__(self, dataset_cls: type, network_fn: Callable, network_args: Dict=None):
        if network_args is None:
            network_args = {}
        self.data = dataset_cls()
        self.network = network_fn(self.data.input_shape, self.data.num_classes, **network_args)
        self.network.summary()

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


if __name__ == '__main__':
    char_model = CharacterModel(EmnistDataset, lenet)
    dataset = EmnistDataset()
    dataset.load_or_generate_data()
    char_model.fit(dataset)
