from typing import Tuple, Union

import numpy as np

from text_recognizer.models import LineModelCtc
import text_recognizer.util as util


class LinePredictor:
    def __init__(self):
        self.model = LineModelCtc()
        self.model.load_weights()
        self.model.network._make_predict_function()    # Bug https://github.com/keras-team/keras/issues/6462

    def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        if isinstance(image_or_filename, str):
                image = util.read_image(image_or_filename, grayscale=True)
        else:
                image = image_or_filename
        return self.model.predict_on_image(image)

    def evaluate(self, dataset):
        return self.model.evaluate(dataset.x_test, dataset.y_test)
