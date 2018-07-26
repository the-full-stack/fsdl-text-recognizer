from collections import defaultdict
import os
import pathlib
from typing import Tuple

import h5py
import nltk
import numpy as np
from tensorflow.keras.utils import to_categorical

from text_recognizer.datasets.base import Dataset
from text_recognizer.datasets.emnist import EmnistDataset
from text_recognizer.datasets.sentences import SentenceGenerator


DATA_DIRNAME = pathlib.Path(__file__).parents[2].resolve() / 'data' / 'processed' / 'emnist_lines'
ESSENTIALS_FILENAME = pathlib.Path(__file__).parents[0].resolve() / 'emnist_lines_essentials.json'


class EmnistLinesDataset(Dataset):
    def __init__(self, max_length: int=32, max_overlap: float=0.4, num_train: int=10000, num_test: int=1000):
        self.emnist = EmnistDataset()
        self.mapping = self.emnist.mapping
        self.max_length = max_length
        self.max_overlap = max_overlap
        self.num_classes = len(self.mapping)
        self.input_shape = (self.emnist.input_shape[0], self.emnist.input_shape[1] * self.max_length)
        self.num_train = num_train
        self.num_test = num_test

    @property
    def data_filename(self):
        return DATA_DIRNAME / f'ml_{self.max_length}_mo{self.max_overlap}_ntr{self.num_train}_nte{self.num_test}.h5'

    def load_or_generate_data(self):
        np.random.seed(42)

        if self.data_filename.exists():
            self._load_data()
        else:
            self._generate_data()
            self._save_data()

    def __repr__(self):
        return (
            'EMNIST Lines Dataset\n'
            f'Max length: {self.max_length}\n'
            f'Max overlap: {self.max_overlap}\n'
            f'Num classes: {self.num_classes}\n'
            f'Input shape: {self.input_shape}\n'
            f'Train: {self.x_train.shape} {self.y_train.shape}\n'
            f'Test: {self.x_test.shape} {self.y_test.shape}\n'
        )

    def _load_data(self):
        print('EmnistLinesDataset loading data from HDF5...')
        with h5py.File(self.data_filename) as f:
            self.x_train = f['x_train'][:]
            self.y_train = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test = f['y_test'][:]

    def _generate_data(self):
        print('EmnistLinesDataset generating data...')
        sentence_generator = SentenceGenerator(self.max_length)

        emnist = self.emnist
        samples_by_char_train = samples_by_char(emnist.x_train, emnist.y_train_int, emnist.mapping)
        samples_by_char_test = samples_by_char(emnist.x_test, emnist.y_test_int, emnist.mapping)

        self.x_train, y_train_str = create_dataset_of_images(self.num_train, samples_by_char_train, sentence_generator, self.max_overlap)
        self.y_train = convert_strings_to_categorical_labels(y_train_str, emnist.inverse_mapping)
        self.x_test, y_test_str = create_dataset_of_images(self.num_test, samples_by_char_test, sentence_generator, self.max_overlap)
        self.y_test = convert_strings_to_categorical_labels(y_test_str, emnist.inverse_mapping)

    def _save_data(self):
        print('EmnistLinesDataset saving data to HDF5...')
        DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.data_filename, 'w') as f:
            f.create_dataset('x_train', data=self.x_train, compression='lzf')
            f.create_dataset('y_train', data=self.y_train, compression='lzf')
            f.create_dataset('x_test', data=self.x_test, compression='lzf')
            f.create_dataset('y_test', data=self.y_test, compression='lzf')


def samples_by_char(samples, labels, mapping):
    samples_by_char = defaultdict(list)
    for sample, label in zip(samples, labels.flatten()):
        samples_by_char[mapping[label]].append(sample)
    return samples_by_char


def select_letter_samples_for_string(string, samples_by_char):
    zero_image = np.zeros((28, 28), 'float32')
    sample_image_by_char = {}
    for char in string:
        if char in sample_image_by_char:
            continue
        samples = samples_by_char[char]
        sample = samples[np.random.choice(len(samples))] if len(samples) > 0 else zero_image
        sample_image_by_char[char] = sample.reshape(28, 28)
    return [sample_image_by_char[char] for char in string]


def construct_image_from_string(string: str, samples_by_char: dict, max_overlap: float) -> np.ndarray:
    overlap = np.random.rand() * max_overlap
    sampled_images = select_letter_samples_for_string(string, samples_by_char)
    N = len(sampled_images)
    H, W = sampled_images[0].shape
    oW = int(overlap * W)
    noW = W - oW
    concatenated_image = np.zeros((H, W * N), 'float32')
    x = 0
    for image in sampled_images:
        concatenated_image[:, x:(x + W)] += image
        x += noW
    return np.minimum(1, concatenated_image)


def create_dataset_of_images(N, samples_by_char, sentence_generator, max_overlap):
    sample_label = sentence_generator.generate()
    sample_image = construct_image_from_string(sample_label, samples_by_char, 0)  # Note that sample_image has 0 overlap
    images = np.zeros((N, sample_image.shape[0], sample_image.shape[1]), 'float32')
    labels = []
    for n in range(N):
        label = sentence_generator.generate()
        images[n] = construct_image_from_string(label, samples_by_char, max_overlap)
        labels.append(label)
    return images, labels


def convert_strings_to_categorical_labels(labels, mapping):
    return np.array([
        to_categorical([mapping[c] for c in label], num_classes=len(mapping))
        for label in labels
    ])


if __name__ == '__main__':
    dataset = EmnistLinesDataset()
    dataset.load_or_generate_data()
    print(dataset)
