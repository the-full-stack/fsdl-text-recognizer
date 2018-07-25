import numpy as np
from tensorflow.keras.utils import Sequence


class DatasetSequence(Sequence):
    """
    Minimal implementation of https://keras.io/utils/#sequence.
    Allows easy use of fit_generator in training.
    """
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        begin = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_x = np.take(self.x, range(begin, end), axis=0, mode='clip')
        batch_y = np.take(self.y, range(begin, end), axis=0, mode='clip')
        return batch_x, batch_y
