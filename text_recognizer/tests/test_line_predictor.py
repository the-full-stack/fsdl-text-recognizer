import editdistance
import pathlib
import unittest

from text_recognizer.line_predictor import LinePredictor
##### Hide lines below until Lab 5
from text_recognizer.datasets import IamLinesDataset
##### Hide lines above until Lab 5


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[0].resolve() / 'support'


class TestEmnistLinePredictor(unittest.TestCase):
    def test_filename(self):
        predictor = LinePredictor()

        for filename in (SUPPORT_DIRNAME / 'emnist_lines').glob('*.png'):
            pred, conf = predictor.predict(str(filename))
            true = filename.stem
            edit_distance = editdistance.eval(pred, true) / len(pred)
            print(f'Pred: "{pred}" | Confidence: {conf} | True: {true} | Edit distance: {edit_distance}')
            self.assertLess(editdistance.eval(pred, filename), 0.2)


##### Hide lines below until Lab 5
class TestIamLinePredictor(unittest.TestCase):
    def test_filename(self):
        predictor = LinePredictor(IamLinesDataset)

        for filename in (SUPPORT_DIRNAME / 'iam_lines').glob('*.png'):
            pred, conf = predictor.predict(str(filename))
            true = filename.stem
            edit_distance = editdistance.eval(pred, true) / len(pred)
            print(f'Pred: "{pred}" | Confidence: {conf} | True: {true} | Edit distance: {edit_distance}')
            self.assertLess(editdistance.eval(pred, filename), 0.2)
##### Hide lines above until Lab 5
