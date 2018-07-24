from typing import Tuple

import numpy as np
import tensorflow as tf

from text_recognizer.datasets.emnist_lines import EmnistLinesDataset
from text_recognizer.models.model import Model


class LineModel(Model):
    def __init__(self):
        np.random.seed(42)
        tf.set_random_seed(42)

        dataset = EmnistLinesDataset()
        self.mapping = dataset.mapping
        self.num_classes = len(self.mapping)
        self.max_length = dataset.max_length
        self.input_shape = dataset.input_shape

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        pred_raw = self.model.predict(np.expand_dims(image, 0), batch_size=1).squeeze()
        pred = convert_pred_raw_to_string(pred_raw, self.mapping)
        conf = np.min(np.max(pred_raw, axis=-1)) # The least confident of the predictions.
        return pred, conf

    def model(self):
        raise NotImplementedError

    loss = 'categorical_crossentropy'


def convert_pred_raw_to_string(preds, mapping):
    return ''.join(mapping[label] for label in np.argmax(preds, axis=-1).flatten()).strip()
