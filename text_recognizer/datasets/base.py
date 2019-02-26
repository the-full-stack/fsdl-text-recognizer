"""Dataset class to be extended by dataset-specific classes."""
import argparse
import pathlib


class Dataset:
    """Simple abstract class for datasets."""
    @classmethod
    def data_dirname(cls):
        return pathlib.Path(__file__).parents[2].resolve() / 'data'

    def load_or_generate_data(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_fraction",
                        type=float,
                        default=None,
                        help="If given, is used as the fraction of data to expose.")
    return parser.parse_args()
