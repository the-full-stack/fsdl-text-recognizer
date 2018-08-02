import pathlib
import unittest

import editdistance
from gradescope_utils.autograder_utils.decorators import weight, leaderboard

from text_recognizer.datasets import EmnistLinesDataset
from text_recognizer.line_predictor import LinePredictor


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[1].resolve() / 'test_support' / 'emnist_lines'


class TestCharacterPredictor(unittest.TestCase):
    @weight(10)
    def test_filename(self):
        predictor = LinePredictor()
        for filename in SUPPORT_DIRNAME.glob('*.png'):
            pred, conf = predictor.predict(str(filename))
            true = filename.stem
            edit_distance = editdistance.eval(pred, true) / len(pred)
            print(f'Pred: "{pred}" | Confidence: {conf} | True: {true} | Edit distance: {edit_distance}')
            self.assertLess(editdistance.eval(pred, filename), 0.2)


class TestEvaluateCharacterPredictor(unittest.TestCase):
    @leaderboard("accuracy")
    def test_evaluate_accuracy(self, set_leaderboard_value=None):
        predictor = LinePredictor()
        dataset = EmnistLinesDataset()
        dataset.load_or_generate_data()
        metric = predictor.evaluate(dataset)
        set_leaderboard_value(metric)
