"""
IAM LINES dataset.
"""
import pathlib

from boltons.cacheutils import cachedproperty
import h5py
from tensorflow.keras.utils import to_categorical

from text_recognizer.datasets.base import Dataset
from text_recognizer.datasets.emnist import EmnistDataset


DATA_DIRNAME = pathlib.Path(__file__).parents[2].resolve() / 'data'
PROCESSED_DATA_DIRNAME = DATA_DIRNAME / 'processed' / 'iam_lines'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'lwitlrt.h5'


class IamLinesDataset(Dataset):
    """
    "The IAM Lines dataset, first published at the ICDAR 1999, contains forms of unconstrained handwritten text,
    which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels.
    From http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

    The data split we will use is
    IAM lines Large Writer Independent Text Line Recognition Task (lwitlrt): 9,862 text lines.
        The validation set has been merged into the train set.
        The train set has 7,101 lines from 326 writers.
        The test set has 1,861 lines from 128 writers.
        The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.

    Note that we use cachedproperty because data takes time to load.
    """
    def __init__(self):
        self.mapping = EmnistDataset().mapping
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.num_classes = len(self.mapping)

    @cachedproperty
    def data(self):
        with h5py.File(PROCESSED_DATA_FILENAME) as f:
            x_train = f['x_train'][:]
            y_train = f['y_train'][:]
            x_test = f['x_test'][:]
            y_test = f['y_test'][:]
        return {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
        }

    @cachedproperty
    def x_train(self):
        return self.data['x_train']

    @cachedproperty
    def x_test(self):
        return self.data['x_test']

    @cachedproperty
    def y_train(self):
        return to_categorical(self.data['y_train'], self.num_classes)

    @cachedproperty
    def y_train_int(self):
        return self.data['y_train']

    @cachedproperty
    def y_test(self):
        return to_categorical(self.data['y_test'], self.num_classes)

    @cachedproperty
    def y_test_int(self):
        return self.data['y_test']

    def __repr__(self):
        return (
            'EMNIST Dataset\n'
            f'Num classes: {self.num_classes}\n'
            f'Mapping: {self.mapping}\n'
            f'Train: {self.x_train.shape} {self.y_train.shape}\n'
            f'Test: {self.x_test.shape} {self.y_test.shape}\n'
        )


if __name__ == '__main__':
    data = IamLinesDataset()
    print(data)
