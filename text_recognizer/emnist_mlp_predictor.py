from typing import Tuple, Union

import numpy as np

from text_recognizer.models.emnist_mlp import EmnistMlp
import text_recognizer.util as util


class EmnistMlpPredictor:
  def __init__(self):
    self.model = EmnistMlp()
    self.model.load_weights()
    self.model.model._make_predict_function()  # Bug https://github.com/keras-team/keras/issues/6462

  def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
    if isinstance(image_or_filename, str):
        image = util.read_image(image_or_filename, grayscale=True)
    else:
        image = image_or_filename
    return self.model.predict_on_image(image)
