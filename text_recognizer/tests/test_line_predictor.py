import editdistance
import pathlib
import unittest
import numpy as np

from text_recognizer.line_predictor import LinePredictor
import text_recognizer.util as util


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[0].resolve() / 'support' / 'emnist_lines'


class TestEmnistLinePredictor(unittest.TestCase):
    def test_filename(self):
        predictor = LinePredictor()

        for filename in SUPPORT_DIRNAME.glob('*.png'):
            pred, conf = predictor.predict(str(filename))
            true = str(filename.stem)
            edit_distance = editdistance.eval(pred, true) / len(pred)
            print(f'Pred: "{pred}" | Confidence: {conf} | True: {true} | Edit distance: {edit_distance}')
            self.assertLess(edit_distance, 0.2)


class TestEmnistLinePredictorVariableImageWidth(unittest.TestCase):
    def test_filename(self):
        predictor = LinePredictor()

        for filename in SUPPORT_DIRNAME.glob('*.png'):
            image = util.read_image(str(filename), grayscale=True)

            print('Saved image shape:', image.shape)
            image = image[:, :-np.random.randint(1, 150)]
            print('Randomly cropped image shape:', image.shape)

            pred, conf = predictor.predict(image)
            true = str(filename.stem)
            edit_distance = editdistance.eval(pred, true) / len(pred)
            print(f'Pred: "{pred}" | Confidence: {conf} | True: {true} | Edit distance: {edit_distance}')
            self.assertLess(edit_distance, 0.2)
