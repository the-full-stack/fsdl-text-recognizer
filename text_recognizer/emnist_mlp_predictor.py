from typing import Tuple

from text_recognizer.models.emnist_mlp import EmnistMlp


class EmnistMlpPredictor:
  def __init__(self):
    self.model = EmnistMlp()
    self.model.load_weights()

  def predict(self, image_filename: str) -> Tuple[str, float]:
    image = util.read_image(image_filename)
    return self.model.predict(image[:])
