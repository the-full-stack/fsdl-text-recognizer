import pathlib
from time import time
import unittest

from text_recognizer.datasets.emnist import EmnistDataset
from text_recognizer.emnist_predictor import EmnistPredictor


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[0].resolve() / 'support' / 'emnist'


class TestEvaluateEmnistPredictor(unittest.TestCase):
    def test_evaluate(self):
        predictor = EmnistPredictor()
        dataset = EmnistDataset()
        t = time()
        metric = predictor.evaluate(dataset)
        time_taken = time() - t
        print(f'acc: {metric}, time_taken: {time_taken}')
        self.assertGreater(metric, 0.8)
        self.assertLess(time_taken, 10)
