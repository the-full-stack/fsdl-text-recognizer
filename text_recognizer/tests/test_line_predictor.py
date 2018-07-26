import editdistance
import pathlib
import unittest

from text_recognizer.line_predictor import LinePredictor


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[0].resolve() / 'support' / 'lines'


class TestEmnistLinePredictor(unittest.TestCase):
    def test_filename(self):
      predictor = LinePredictor()

      for filename in SUPPORT_DIRNAME.glob('*.png'):
        pred, conf = predictor.predict(str(filename))
        print(pred, conf, filename.stem)
        self.assertEqual(pred, filename.stem)
        self.assertGreater(conf, 0.9)
