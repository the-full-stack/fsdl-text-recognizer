import numpy as np
from tensorflow.keras.utils import Sequence


class DatasetSequence(Sequence):
    """
    Minimal implementation of https://keras.io/utils/#sequence.
    Allows easy use of fit_generator in training.
    """
    def __init__(self, x, y, batch_size, augment_fn=None, format_fn=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augment_fn = augment_fn
        self.format_fn = format_fn

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        begin = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        batch_x = np.take(self.x, range(begin, end), axis=0, mode='clip')
        batch_y = np.take(self.y, range(begin, end), axis=0, mode='clip')

        if self.augment_fn:
            batch_x, batch_y = self.augment_fn(batch_x, batch_y)

        if self.format_fn:
            batch_inputs, batch_outputs = self.format_fn(batch_x, batch_y)

        return batch_inputs, batch_outputs
