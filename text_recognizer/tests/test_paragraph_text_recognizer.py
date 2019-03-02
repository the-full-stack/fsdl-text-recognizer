"""Tests for ParagraphTextRecognizer class."""
import os
from pathlib import Path
import unittest
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util


SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / 'support' / 'iam_paragraphs'

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestEmnistLinePredictor(unittest.TestCase):
    def test_filename(self):  # pylint: disable=R0201
        predictor = ParagraphTextRecognizer()
        text_offset = 300
        for filename in (SUPPORT_DIRNAME).glob('*.jpg'):
            
            full_image = util.read_image(str(filename), grayscale=True)
            roi_image = full_image[text_offset:, :]
            
            predicted_text, line_region_crops = predictor.predict(roi_image)
            print(predicted_text)
            print(len(line_region_crops))
            assert len(line_region_crops) == 7
