import pathlib
from typing import Callable, Dict, Tuple

from boltons.cacheutils import cachedproperty
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

from text_recognizer.models.base import Model
from text_recognizer.datasets.emnist import EmnistDataset
from text_recognizer.networks.mlp import mlp
from text_recognizer.networks.lenet import lenet


class CharacterModel(Model):
    def __init__(self, dataset_cls: type=EmnistDataset, network_fn: Callable=mlp, network_args: Dict=None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, network_args)

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        # Your code below (Lab 1)
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        conf = pred_raw[ind]
        pred = self.data.mapping[ind]
        # Your code above (Lab 1)
        return pred, conf
