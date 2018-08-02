import pathlib
import unittest

from gradescope_utils.autograder_utils.decorators import weight, leaderboard

from text_recognizer.datasets import EmnistDataset
from text_recognizer.character_predictor import CharacterPredictor


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[1].resolve() / 'test_support' / 'emnist'


class TestCharacterPredictor(unittest.TestCase):
    @weight(10)
    def test_filename(self):
      predictor = CharacterPredictor()

      for filename in SUPPORT_DIRNAME.glob('*.png'):
        pred, conf = predictor.predict(str(filename))
        print(pred, conf, filename.stem)
        self.assertEqual(pred, filename.stem)


class TestEvaluateCharacterPredictor(unittest.TestCase):
    @leaderboard("accuracy")
    def test_evaluate_accuracy(self, set_leaderboard_value=None):
        predictor = CharacterPredictor()
        dataset = EmnistDataset()
        dataset.load_or_generate_data()
        metric = predictor.evaluate(dataset)
        set_leaderboard_value(metric)
