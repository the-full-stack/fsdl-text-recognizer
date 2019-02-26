"""Dataset class to be extended by dataset-specific classes."""
from pathlib import Path
from typing import Union
import argparse
import hashlib


class Dataset:
    """Simple abstract class for datasets."""
    @classmethod
    def data_dirname(cls):
        return Path(__file__).parents[2].resolve() / 'data'

    def load_or_generate_data(self):
        pass


def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_fraction",
                        type=float,
                        default=None,
                        help="If given, is used as the fraction of data to expose.")
    return parser.parse_args()
