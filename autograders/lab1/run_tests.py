import pathlib
import unittest

from gradescope_utils.autograder_utils.decorators import weight, leaderboard
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

from text_recognizer.datasets import EmnistDataset
from text_recognizer.emnist_predictor import EmnistPredictor
from text_recognizer.models.emnist_mlp import EmnistMlp


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[0].resolve() / 'tests' / 'support'


class TestEmnistPredictor(unittest.TestCase):
    @weight(10)
    def test_filename(self):
      predictor = EmnistPredictor()

      for filename in SUPPORT_DIRNAME.glob('*.png'):
        pred, conf = predictor.predict(str(filename))
        print(pred, conf, filename.stem)
        self.assertEqual(pred, filename.stem)
        self.assertGreater(conf, 0.4)


class TestEvaluateEmnistPredictor(unittest.TestCase):
    @leaderboard("accuracy")
    def test_evaluate_accuracy(self, set_leaderboard_value=None):
        dataset = EmnistDataset()
        model = EmnistMlp()
        metric = model.evaluate(dataset.x_test, dataset.y_test)
        set_leaderboard_value(metric)


if __name__ == '__main__':
    JSONTestRunner(visibility='visible').run(unittest.main())
