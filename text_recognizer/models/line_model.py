from typing import Callable, Dict, Tuple

import numpy as np
import tensorflow as tf

from text_recognizer.datasets.emnist_lines import EmnistLinesDataset
from text_recognizer.models.base import Model
from text_recognizer.networks import line_cnn_sliding_window


class LineModel(Model):
    def __init__(self, dataset_cls: type=EmnistLinesDataset, network_fn: Callable=line_cnn_sliding_window, network_args: Dict=None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, network_args)

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).squeeze()
        pred = convert_pred_raw_to_string(pred_raw, self.data.mapping)
        conf = np.min(np.max(pred_raw, axis=-1)) # The least confident of the predictions.
        return pred, conf

    loss = 'categorical_crossentropy'


def convert_pred_raw_to_string(preds, mapping):
    return ''.join(mapping[label] for label in np.argmax(preds, axis=-1).flatten()).strip()
