import pathlib
import unittest

from text_recognizer.predict.emnist_mlp import EmnistMlpPredictor


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[0].resolve() / 'support' / 'emnist'


class TestEmnistMlpPredictor(unittest.TestCase):
    def test():
      predictor = EmnistMlpPredictor()

      filename = SUPPORT_DIRNAME / 'a.png'
      pred, conf = predictor.predict(filename)
      self.assertEqual(pred, 'a')
      self.assertGreater(conf, 0.9)
