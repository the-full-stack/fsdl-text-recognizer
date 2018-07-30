import pathlib
import shutil

import numpy as np

from text_recognizer.datasets.emnist import EmnistDataset
import text_recognizer.util as util


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[0].resolve() / 'emnist'


def create_emnist_support_files():
    shutil.rmtree(SUPPORT_DIRNAME, ignore_errors=True)
    SUPPORT_DIRNAME.mkdir()

    dataset = EmnistDataset()

    for ind in [1, 2, 4]:
        image = (dataset.x_test[ind] * 255).astype(np.uint8)
        label = dataset.mapping[np.argmax(dataset.y_test[ind])]
        util.write_image(image, str(SUPPORT_DIRNAME / f'{label}.png'))


if __name__ == '__main__':
    create_emnist_support_files()
