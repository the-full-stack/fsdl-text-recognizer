"""
Takes an image and returns all the text in it, by first segmenting the image with LineDetector, then extracting crops
of the image corresponding to the line regions, and running each line region crop through LinePredictor.
"""
from typing import List, Tuple, Union
import cv2
import numpy as np
from text_recognizer.models.line_detector_model import LineDetectorModel
from text_recognizer.models.line_model_ctc import LineModelCtc
import text_recognizer.util as util


class ParagraphTextRecognizer:
    """Given an image of a single handwritten character, recognizes it."""
    def __init__(self):
        self.line_detector_model = LineDetectorModel()
        self.line_detector_model.load_weights()
        # self.line_predictor_model = LineModelCtc()
        # self.line_predictor_model.load_weights()

    def predict(self, image_or_filename: Union[np.ndarray, str]):
        """
        Takes an image and returns all the text in it.
        """
        if isinstance(image_or_filename, str):
            image = util.read_image(image_or_filename, grayscale=True)
        else:
            image = image_or_filename

        square_image = _crop_out_square_image(image)

        line_region_crops = self._get_line_region_crops(square_image)
        return line_region_crops
        # resize down image to height = 28?
        # line_region_strings = [self.line_predictor_model.predict(crop) for crop in line_region_crops]
        # return ' '.join(line_region_strings)

    def _get_line_region_crops(self, square_image: np.ndarray) -> List[np.ndarray]:
        image, scale_down_factor = _resize_image(
            image=square_image,
            expected_shape=self.line_detector_model.image_shape
        )
        image = _format_image_for_line_detector_model(image)

        line_segmentation = self.line_detector_model.predict_on_image(image)
        bounding_boxes_wyxh = _find_line_bounding_boxes(line_segmentation)

        bounding_boxes_wyxh = (bounding_boxes_wyxh * scale_down_factor).astype(int)
        line_region_crops = [square_image[y:y+h, x:x+w] for x, y, w, h in bounding_boxes_wyxh]
        return line_region_crops


def _find_line_bounding_boxes(line_segmentation: np.ndarray):
    """Given a line segmentation, find bounding boxes for connected-component regions corresponding to non-0 labels."""

    def _find_line_bounding_boxes_in_channel(line_segmentation_channel: np.ndarray) -> np.ndarray:
        line_activation_image = cv2.dilate(line_segmentation_channel, kernel=np.ones((3, 3)), iterations=1)
        line_activation_image = (line_activation_image * 255).astype('uint8')
        line_activation_image = cv2.threshold(line_activation_image, 0.5, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        _, bounding_cnts, _ = cv2.findContours(line_activation_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return np.array([cv2.boundingRect(cnt) for cnt in bounding_cnts])

    bboxes_xywh = np.concatenate([
        _find_line_bounding_boxes_in_channel(line_segmentation[:, :, i])
        for i in [1, 2]
    ], axis=0)
    return bboxes_xywh[np.argsort(bboxes_xywh[:, 1])]


def _crop_out_square_image(image: np.ndarray) -> np.ndarray:
    """Crop out the largest square from the image at the center."""
    if image.shape[0] == image.shape[1]:
        return image.copy()
    image_shape = np.array(image.shape)
    crop_len = image_shape.min()
    crop_axis = image_shape.argmax()
    if crop_axis == 0:
        y1, y2 = image_shape[crop_axis] // 2 - crop_len // 2, image_shape[crop_axis] // 2 + crop_len // 2
        x1, x2 = 0, image_shape[1]
    else:
        x1, x2 = image_shape[crop_axis] // 2 - crop_len // 2, image_shape[crop_axis] // 2 + crop_len // 2
        y1, y2 = 0, image_shape[0]

    return image[y1:y2, x1:x2]


def _resize_image(image: np.ndarray, expected_shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """If the image is of expected_shape shape, then crop the center, and resize it to the expected_shape."""
    assert image.shape[0] == image.shape[1]
    if image.shape == expected_shape:
        return image.copy(), 1.
    scale_down_factor = image.shape[0] / expected_shape[0]
    return cv2.resize(image, dsize=expected_shape, interpolation=cv2.INTER_AREA), scale_down_factor


def _format_image_for_line_detector_model(image: np.ndarray) -> np.ndarray:
    """Convert uint8 image to float image with black background."""
    return (1. - image / 255.).astype('float32')
