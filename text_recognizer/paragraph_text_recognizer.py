"""
Takes an image and returns all the text in it, by first segmenting the image with LineDetector, then extracting crops
of the image corresponding to the line regions, and running each line region crop through LinePredictor.
"""
from typing import Tuple, Union
import numpy as np
from text_recognizer.models.line_detector_model import LineDetectorModel
from text_recognizer.models.line_model_ctc import LineModelCtc
import text_recognizer.util as util


class ParagraphTextRecognizer:
    """Given an image of a single handwritten character, recognizes it."""
    def __init__(self):
        self.line_detector_model = LineDetectorModel()
        self.line_detector_model.load_weights()
        self.line_predictor_model = LineModelCtc()
        self.line_predictor_model.load_weights()

    def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """
        Takes an image and returns all the text in it.
        """
        if isinstance(image_or_filename, str):
            image = util.read_image(image_or_filename, grayscale=True)
        else:
            image = image_or_filename

        image_for_line_detection = self.crop_and_resize_image_for_line_detection(image)
        segmentation = self.line_detector_model.predict_on_image(image_for_line_detection)
        line_region_crops = _extract_crops_from_segmentation(image, segmentation)
        line_region_strings = [self.line_predictor_model.predict(crop) for crop in line_region_crops]
        return ' '.join(line_region_strings)

    def crop_and_resize_image_for_line_detection(self, image: np.ndarray) -> np.ndarray:
        """
        If the image is not the size that self.line_detector_model expects, then crop the center, and resize it to the
        correct size.
        """
        expected_shape = self.line_detector_model.image_shape
        return image


def _extract_crops_from_segmentation(image: np.ndarray, segmentation: np.ndarray):
    """
    Given a segmentation, extract connected-component regions corresponding to non-0 labels  from the image.
    """
    # TODO
