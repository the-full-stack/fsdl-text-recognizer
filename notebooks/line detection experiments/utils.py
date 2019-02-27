import os
import random
import string
from typing import Dict, List
import matplotlib as mpl
# pylint: disable=C0413
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa
from matplotlib.figure import Figure  # noqa


_IDENTIFIER_LENGTH = 8


def plot_training_history(history: Dict[str, List[float]], metric: str = 'acc', filename: str = '') -> Figure:
    """Plot keras training history in a two-pane display of loss and metric."""
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(121)
    ax.plot(history['loss'], label='train loss')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    if metric is not None:
        ax2 = fig.add_subplot(122)
        ax2.plot(history[metric], label='train {}'.format(metric))
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax2.plot(history[val_metric], label='val {}'.format(metric))
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric)
        ax2.legend()
    if filename:
        plt.savefig(filename)


def generate_unique_identifier(dirname_with_identifiers: str, identifier_length: int = _IDENTIFIER_LENGTH) -> str:
    """
    dirname_with_identifiers is the path to the directory with sub-directories created with previous identifiers.
    This function generates and returns an identifier which is not already present in dirname_with_identifiers.
    """
    if os.path.exists(dirname_with_identifiers):
        existing_identifiers = os.listdir(dirname_with_identifiers)
    else:
        existing_identifiers = []

    while 1:
        identifier = ''.join(random.choices(string.ascii_lowercase + string.digits, k=identifier_length))
        if identifier not in existing_identifiers:
            break
    return identifier        