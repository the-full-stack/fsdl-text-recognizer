"""Emnist Lines dataset: synthetic handwriting lines dataset made from EMNIST characters."""
from collections import defaultdict
from pathlib import Path
from typing import List

import h5py
import numpy as np
from boltons.cacheutils import cachedproperty
from tensorflow.keras.utils import to_categorical

from text_recognizer.datasets.dataset import Dataset
from text_recognizer.datasets.emnist_dataset import EmnistDataset


DATA_DIRNAME = Dataset.data_dirname() / "processed" / "emnist_lines"
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_lines_essentials.json"


class EmnistLinesDataset(Dataset):
    """
    EmnistLinesDataset class.

    Parameters
    ----------
    max_length
        Max line length in characters.
    max_overlap
        Max overlap between characters in a line.
    num_train
        Number of training examples to generate.
    num_test
        Number of test examples to generate.
    categorical_format
        If True, then y labels are given as one-hot vectors.
    with_start_and_end_tokens
        If True, start and end each sequence with special tokens.
    """

    def __init__(
        self,
        max_length: int = 34,
        min_overlap: float = 0,
        max_overlap: float = 0.33,
        num_train: int = 100000,
        num_test: int = 10000,
        categorical_format: bool = False,
        with_start_and_end_labels: bool = False,
    ):
        self.categorical_format = categorical_format
        self.with_start_and_end_labels = with_start_and_end_labels

        self.emnist = EmnistDataset()

        self.mapping = _augment_mapping(self.emnist.mapping)
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.padding_label = self.inverse_mapping["_"]
        self.start_label = self.inverse_mapping["<s>"]
        self.end_label = self.inverse_mapping["<e>"]

        self.max_length = max_length
        self.max_output_length = self.max_length
        if self.with_start_and_end_labels:
            self.max_output_length += 2

        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.num_classes = len(self.mapping)
        self.input_shape = (
            self.emnist.input_shape[0],
            self.emnist.input_shape[1] * self.max_length,
        )
        self.output_shape = (self.max_output_length, self.num_classes)
        self.num_train = num_train
        self.num_test = num_test

        self.x_train = None
        self.y_train_int = None
        self.x_test = None
        self.y_test_int = None

    @property
    def data_filename(self):
        name = f"ml_{self.max_length}_o{self.min_overlap}_{self.max_overlap}_ntr{self.num_train}_nte{self.num_test}.h5"
        return DATA_DIRNAME / name

    def load_or_generate_data(self):
        np.random.seed(42)

        if not self.data_filename.exists():
            self._generate_data("train")
            self._generate_data("test")
        self._load_data()

    @cachedproperty
    def y_train(self):
        return self.format_y_int(self.y_train_int)

    @cachedproperty
    def y_test(self):
        return self.format_y_int(self.y_test_int)

    def format_y_int(self, y):
        if self.with_start_and_end_labels:
            y = add_start_and_end_labels(y, self.padding_label, self.start_label, self.end_label)
        if self.categorical_format:
            y = to_categorical(y, self.num_classes)
        return y

    def __repr__(self):
        return (
            "EMNIST Lines Dataset\n"  # pylint: disable=no-member
            f"Max length: {self.max_length}\n"
            f"Min overlap: {self.min_overlap}\n"
            f"Max overlap: {self.max_overlap}\n"
            f"Num classes: {self.num_classes}\n"
            f"Input shape: {self.input_shape}\n"
            f"Train: {self.x_train.shape} {self.y_train.shape}\n"
            f"Test: {self.x_test.shape} {self.y_test.shape}\n"
        )

    def _load_data(self):
        print("EmnistLinesDataset loading data from HDF5...")
        with h5py.File(self.data_filename, "r") as f:
            self.x_train = f["x_train"][:]
            self.y_train_int = f["y_train_int"][:]
            self.x_test = f["x_test"][:]
            self.y_test_int = f["y_test_int"][:]

    def _generate_data(self, split):
        print("EmnistLinesDataset generating data...")

        # pylint: disable=import-outside-toplevel
        from text_recognizer.datasets.sentence_generator import SentenceGenerator

        sentence_generator = SentenceGenerator(self.max_length)

        emnist = self.emnist
        emnist.load_or_generate_data()
        if split == "train":
            samples_by_char = get_samples_by_char(emnist.x_train, emnist.y_train_int, emnist.mapping)
        else:
            samples_by_char = get_samples_by_char(emnist.x_test, emnist.y_test_int, emnist.mapping)

        num = self.num_train if split == "train" else self.num_test

        DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.data_filename, "a") as f:
            x, y = create_dataset_of_images(
                num, samples_by_char, sentence_generator, self.min_overlap, self.max_overlap
            )
            y = convert_strings_to_ints(y, self.inverse_mapping)
            f.create_dataset(f"x_{split}", data=x, dtype="u1", compression="lzf")
            f.create_dataset(f"y_{split}_int", data=y, dtype="u1", compression="lzf")


def get_samples_by_char(samples, labels, mapping):
    samples_by_char = defaultdict(list)
    for sample, label in zip(samples, labels.flatten()):
        samples_by_char[mapping[label]].append(sample)
    return samples_by_char


def select_letter_samples_for_string(string, samples_by_char):
    zero_image = np.zeros((28, 28), np.uint8)
    sample_image_by_char = {}
    for char in string:
        if char in sample_image_by_char:
            continue
        samples = samples_by_char[char]
        sample = samples[np.random.choice(len(samples))] if samples else zero_image
        sample_image_by_char[char] = sample.reshape(28, 28)
    return [sample_image_by_char[char] for char in string]


def construct_image_from_string(
    string: str, samples_by_char: dict, min_overlap: float, max_overlap: float
) -> np.ndarray:
    overlap = np.random.uniform(min_overlap, max_overlap)
    sampled_images = select_letter_samples_for_string(string, samples_by_char)
    N = len(sampled_images)
    H, W = sampled_images[0].shape
    next_overlap_width = W - int(overlap * W)
    concatenated_image = np.zeros((H, W * N), np.uint8)
    x = 0
    for image in sampled_images:
        concatenated_image[:, x : (x + W)] += image
        x += next_overlap_width
    return np.minimum(255, concatenated_image)


def create_dataset_of_images(N, samples_by_char, sentence_generator, min_overlap, max_overlap):
    sample_label = sentence_generator.generate()
    sample_image = construct_image_from_string(sample_label, samples_by_char, 0, 0)  # sample_image has 0 overlap
    images = np.zeros(
        (N, sample_image.shape[0], sample_image.shape[1]), np.uint8,  # pylint: disable=unsubscriptable-object
    )
    labels = []
    for n in range(N):
        label = None
        for _ in range(10):  # Try several times to generate before actually erroring
            try:
                label = sentence_generator.generate()
                break
            except Exception:  # pylint: disable=broad-except
                pass
        if label is None:
            raise RuntimeError("Was not able to generate a valid string")
        images[n] = construct_image_from_string(label, samples_by_char, min_overlap, max_overlap)
        labels.append(label)
    return images, labels


def convert_strings_to_ints(labels: List[str], mapping: dict) -> np.ndarray:
    return np.array([[mapping[c] for c in label] for label in labels])


def _augment_mapping(mapping):
    """Augment the character mapping with punctuation and with padding, start, and end tokens."""
    # Extra symbols in IAM dataset
    extra_symbols = [
        " ",
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
    ]

    # Special padding label
    extra_symbols.append("_")

    # Special start and end labels
    extra_symbols.append("<s>")
    extra_symbols.append("<e>")

    max_key = max(mapping.keys())
    extra_mapping = {}
    for i, symbol in enumerate(extra_symbols):
        extra_mapping[max_key + 1 + i] = symbol

    return {**mapping, **extra_mapping}


def add_start_and_end_labels(y, padding_label, start_label, end_label):
    N = y.shape[0]
    y = np.hstack((np.ones((N, 1)) * start_label, y, np.ones((N, 1)) * padding_label))
    y[range(N), [np.where(row == padding_label)[0][0] for row in y]] = end_label
    return y


def main():
    dataset = EmnistLinesDataset()
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == "__main__":
    main()
