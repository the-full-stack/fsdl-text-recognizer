import pathlib

import numpy as np

from text_recognizer.datasets.emnist import Emnist
import text_recognizer.util as util


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[0].resolve() / 'emnist'


def create_emnist_support_files():
    SUPPORT_DIRNAME.mkdir(parents=True, exist_ok=True)
    # TODO: delete all existing files in the directory

    data = EmnistDataset()

    for ind in [1, 2, 4]:
        image = (data.x_test[ind].reshape(28, 28) * 255).astype(np.uint8).T
        label = data.mapping[np.argmax(data.y_test[ind])]
        util.write_image(image, str(SUPPORT_DIRNAME / f'{label}.png'))


if __name__ == '__main__':
    create_emnist_support_files()
