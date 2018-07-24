"""
EMNIST dataset. Downloads from NIST website and saves as .npz file if not already present.
"""
import json
import os
import pathlib
import shutil
import urllib.request
import zipfile

from boltons.cacheutils import cachedproperty
import h5py
import numpy as np
from tensorflow.python.keras.utils import to_categorical


RAW_URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'

DATA_DIRNAME = pathlib.Path(__file__).parents[2].resolve() / 'data'
RAW_DATA_DIRNAME = DATA_DIRNAME / 'raw' / 'emnist'
PROCESSED_DATA_DIRNAME = DATA_DIRNAME / 'processed' / 'emnist'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'byclass.h5'
ESSENTIALS_FILENAME = pathlib.Path(__file__).parents[0].resolve() / 'emnist_essentials.json'


def _download_and_process_emnist():
    import scipy.io

    RAW_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)

    os.chdir(RAW_DATA_DIRNAME)

    if not os.path.exists('matlab.zip'):
        print('Downloading EMNIST...')
        urllib.request.urlretrieve(RAW_URL, 'matlab.zip')

    print('Unzipping EMNIST and loading .mat file...')
    zip_file = zipfile.ZipFile('matlab.zip', 'r')
    zip_file.extract('matlab/emnist-byclass.mat', )
    data = scipy.io.loadmat('matlab/emnist-byclass.mat')

    print('Saving to HDF5...')
    x_train = data['dataset']['train'][0, 0]['images'][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_train = data['dataset']['train'][0, 0]['labels'][0, 0]
    x_test = data['dataset']['test'][0, 0]['images'][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_test = data['dataset']['test'][0, 0]['labels'][0, 0]
    with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
        f.create_dataset('x_train', data=x_train, compression='lzf')
        f.create_dataset('y_train', data=y_train, compression='lzf')
        f.create_dataset('x_test', data=x_test, compression='lzf')
        f.create_dataset('y_test', data=y_test, compression='lzf')

    print('Saving essential dataset parameters...')
    mapping = {int(k): chr(v) for k, v in data['dataset']['mapping'][0, 0]}
    essentials = {'mapping': list(mapping.items()), 'input_shape': list(x_train.shape[1:])}
    with open(ESSENTIALS_FILENAME, 'w') as f:
        json.dump(essentials, f)

    print('Cleaning up...')
    shutil.rmtree('matlab')

    print('EMNIST downloaded and processed')


def _augment_emnist_mapping(mapping):
    """
    We should augment the mapping with three extra characters:

    - ' ' for space between words
    - '_' for padding around the line
    - '?' for all unknown characters
    """
    max_key = max(mapping.keys())
    extra_mapping = {
        max_key + 1: ' ',
        max_key + 2: '?',
        max_key + 3: '_'
    }
    return {**mapping, **extra_mapping}


class EmnistDataset(object):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.

    Note that we use cachedproperty because data takes time to load.
    """
    def __init__(self):
        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)
        self.mapping = _augment_emnist_mapping(dict(essentials['mapping']))
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.num_classes = len(self.mapping)
        self.input_shape = essentials['input_shape']

    @cachedproperty
    def data(self):
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_emnist()
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
        return self.data['x_train'].astype('float32') / 255

    @cachedproperty
    def x_test(self):
        return self.data['x_test'].astype('float32') / 255

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
            f'Input shape: {self.input_shape}\n'
            f'Train: {self.x_train.shape} {self.y_train.shape}\n'
            f'Test: {self.x_test.shape} {self.y_test.shape}\n'
        )


if __name__ == '__main__':
    data = EmnistDataset()
    print(data)
