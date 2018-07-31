"""Module to encapsulate text region extraction implementation."""
from typing import List, Dict, Tuple, Union, TypeVar
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
# from scipy.ndimage import label
import util


Bbox = Dict[str, int]  # pylint: disable=C0103
IntFloat = TypeVar('IntFloat', int, float)  # pylint: disable=C0103

_DEFAULT_PARAMS: Dict[str, IntFloat] = {
    'aspect_ratio_percentage_cutoff': 3,
    'min_height_cutoff': 50,
    'max_height_cutoff': 400,
    'discard_area_percentage_cutoff': 0.12,
    'discard_area_absolute_cutoff': 2.3e3,
    'discard_text_percentage_cutoff': 0.025,
    'merge_area_percentage_cutoff': 0.30,
    'merge_area_absolute_cutoff': 3500,
    'merge_distance_cutoff': 40,
    'perform_trimming': 1,
    'dilation_kernel_length': 100,
    'gaussian_blur_size': 21,
    'concatenation_padding': 150
}


def discard_tall_bboxes(bboxes, max_height_cutoff: float):
    return [bbox for bbox in bboxes if bbox['y2'] - bbox['y1'] < max_height_cutoff]


def threshold_text_image(image: np.ndarray, blur_kernel_size: int = 21, threshold_window_size: int = 51) -> np.ndarray:
    """Performs a blur to remove noise and thresholds the image to binarize it."""
    img = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    thresh_sauvola = threshold_sauvola(img, window_size=threshold_window_size)
    return ((img > thresh_sauvola) * 255).astype('uint8')


def get_text_span(white_text_image, axis, pixel_intensity_noise_threshold: float = 0.05):
    """
    Returns the span of text in the white_text_image along the axis.
    """
    img_sum = np.sum(white_text_image, axis=axis)
    img_sum = img_sum / img_sum.max()

    img_sum = cv2.GaussianBlur(img_sum, (1, 101), 0)
    img_sum = img_sum / img_sum.max()

    img_sum[img_sum <= pixel_intensity_noise_threshold] = 0
    text_span = len(np.where(img_sum > 0)[0])

    # labels, num_lines = label(img_sum)
    # line_widths = [len(np.where(labels == i+1)[0]) for i in range(num_lines)]
    # return text_span, num_lines, line_widths
    return text_span


def horizontally_dilate_image(img: np.ndarray, dilation_kernel_length: int) -> np.ndarray:
    """Dilates with a long horizontal structuring element to bleed text together."""
    long_horizontal_kernel = np.ones((3, dilation_kernel_length), np.uint8)
    return cv2.dilate(img, long_horizontal_kernel, iterations=1)


def extract_text_regions(white_text_image: np.ndarray,
                         merge_fully_overlapping_regions: bool = True,
                         params = None,
                         merge_bboxes_in_line: bool = True):
    """
    Takes in a thresholded image with white text on black background, and returns the list of region images
    and their corresponding bounding boxes.
    """
    if not params:
        params = _DEFAULT_PARAMS

    preprocessed_image = horizontally_dilate_image(white_text_image,
                                                   dilation_kernel_length=int(params['dilation_kernel_length']))
    # Find contours and initial bboxes
    bboxes, _ = find_bboxes(preprocessed_image)

    # Shrinks the width of the bboxes due to initial horizontal dilation
    if params['perform_trimming']:
        bboxes = trim_bboxes(bboxes, white_text_image.shape[1])

    bboxes = discard_tall_bboxes(bboxes, params['max_height_cutoff'])

    # Merges nearby bboxes
    bboxes = merge_close_bboxes(bboxes,
                                merge_area_percentage_cutoff=params['merge_area_percentage_cutoff'],
                                merge_area_absolute_cutoff=params['merge_area_absolute_cutoff'],
                                merge_distance_cutoff=params['merge_distance_cutoff'],
                                merge_fully_overlapping_regions=merge_fully_overlapping_regions)

    # Discards regions with little text (small percentage of white pixels)
    regions = bboxes_to_regions(bboxes, white_text_image)
    region_bbox_pairs = discard_regions_with_little_text(
        list(zip(regions, bboxes)),
        discard_text_percentage_cutoff=params['discard_text_percentage_cutoff'],
        discard_area_percentage_cutoff=params['discard_area_percentage_cutoff'],
        discard_area_absolute_cutoff=params['discard_area_absolute_cutoff']
    )
    if region_bbox_pairs:
        regions, bboxes = zip(*region_bbox_pairs)
    else:
        regions, bboxes = [], []

    # Discards regions with aspect ratio that does not resemble a horizontal line
    bboxes = discard_bboxes_with_bad_aspect_ratio(
        bboxes,
        aspect_ratio_percentage_cutoff=params['aspect_ratio_percentage_cutoff'],
        min_height_cutoff=params['min_height_cutoff']
    )

    if merge_bboxes_in_line:
        bboxes_per_line = decompose_into_lines(bboxes)
        bboxes = [merge_bboxes(bboxes) for bboxes in bboxes_per_line]
        bboxes.sort(key=lambda line_bbox: line_bbox['y1'])

    # Recalculates regions based on bboxes
    regions = bboxes_to_regions(bboxes, white_text_image)
    return list(regions), list(bboxes)


def find_bboxes(img: np.ndarray) -> Tuple[List[Bbox], List[np.ndarray]]:
    """Takes a preprocessed image and returns the contours and bboxes found in the image"""
    _, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return list(map(lambda contour: correct_bbox_format(cv2.boundingRect(contour)), contours)), contours


def correct_bbox_format(bbox: List[int]) -> Bbox:
    """
    Takes a bbox from the output of cv2.boundingRect
    that is a tuple of (x1, y1, w, h) and converts them to
    a dict of x1, y1, x2, y2
    """
    return {
        'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[0] + bbox[2], 'y2': bbox[1] + bbox[3]
    }


def trim_bboxes(bboxes: List[Bbox], img_width: int) -> List[Bbox]:
    """calls trimmer() for each bbox in bboxes."""
    def trimmer(bbox):
        """
        Cuts off 48px from both ends of the line, to compensate for dilation
        Does not trim the ends of bboxes that are adjacent to the ends of an image,
        because these were probably not affected much by dilation
        """
        x1, x2 = bbox['x1'], bbox['x2']
        if bbox['x2'] < img_width:
            x2 = max(int((bbox['x2'] + bbox['x1']) / 2) + 1, bbox['x2'] - 45)
        if bbox['x1'] > 0:
            x1 = min(int((bbox['x2'] + bbox['x1']) / 2) - 1, bbox['x1'] + 45)
        return {
            'x1': x1,
            'y1': bbox['y1'],
            'x2': x2,
            'y2': bbox['y2'],
        }
    return list(map(trimmer, bboxes))


def merge_close_bboxes(bboxes: List[Bbox],
                       merge_area_percentage_cutoff: float,
                       merge_area_absolute_cutoff: float,
                       merge_distance_cutoff: float,
                       merge_fully_overlapping_regions: bool) -> List[Bbox]:
    """
    Merges smaller bboxes with larger ones. This is meant to merge the bottom parts
    of letters like 'y' and 'g', and the top parts of letters like 'd' and 'h'
    Only merges 2 bboxes if they are closer then merge_distance_cutoff, and if the
    smaller one is less than merge_area_percentage cutoff factor of the larger one.
    """
    sorted_by_area = sorted(bboxes, key=area_of_bbox)
    num_bboxes = len(sorted_by_area)
    merge_cache = []
    for index in range(len(sorted_by_area) - 1):
        bbox = sorted_by_area[index]
        closest_distance, closest_index = find_closest_bbox_index(bbox, sorted_by_area[index + 1:][::-1])
        if closest_distance <= merge_distance_cutoff:
            merge_cache.append((index, num_bboxes - 1 - closest_index, closest_distance))

    for item in merge_cache:
        small_bbox, big_bbox = sorted_by_area[item[0]], sorted_by_area[item[1]]
        distance_between_bboxes = item[2]
        if (area_of_bbox(small_bbox) / area_of_bbox(big_bbox) < merge_area_percentage_cutoff and
                area_of_bbox(small_bbox) < merge_area_absolute_cutoff) or \
           (distance_between_bboxes == 0 and merge_fully_overlapping_regions):
            merged_bbox = merge_bboxes([small_bbox, big_bbox])
            sorted_by_area[item[0]], sorted_by_area[item[1]] = None, merged_bbox
    return list(filter(lambda x: x is not None, sorted_by_area))


def area_of_bbox(bbox: Bbox) -> int:
    """Takes a bbox and returns its area"""
    return (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])


def find_closest_bbox_index(bbox: Bbox, bboxes: List[Bbox]) -> Tuple[float, int]:
    """
    Returns index of element in bboxes which is closest to bbox,
    measured by the distance between boundaries
    """
    distances = list(map(lambda bbox2: bbox_distance(bbox, bbox2), bboxes))
    closest_distance = min(distances)
    closest_index = distances.index(closest_distance)
    return closest_distance, closest_index


def bbox_distance(bbox1: Bbox, bbox2: Bbox) -> float:  # pylint: disable=R0911
    """
    Returns the distance between borders of bboxes. In case of any partial overlap, 0.5 is returned and in case of full
    overlap 0 is returned. Assumes that x2 > x1 and y2 > y1.
    """
    if bbox1['x2'] < bbox2['x1'] and bbox1['y2'] < bbox2['y1']:
        return max(bbox2['x1'] - bbox1['x2'], bbox2['y1'] - bbox1['y2'])
    elif bbox1['x2'] < bbox2['x1'] and bbox1['y1'] > bbox2['y2']:
        return max(bbox2['x1'] - bbox1['x2'], bbox1['y1'] - bbox2['y2'])
    elif bbox1['x1'] > bbox2['x2'] and bbox1['y2'] < bbox2['y1']:
        return max(bbox1['x1'] - bbox2['x2'], bbox2['y1'] - bbox1['y2'])
    elif bbox1['x1'] > bbox2['x2'] and bbox1['y1'] > bbox2['y2']:
        return max(bbox1['x1'] - bbox2['x2'], bbox1['y1'] - bbox2['y2'])
    elif bbox1['x2'] < bbox2['x1']:
        return bbox2['x1'] - bbox1['x2']
    elif bbox1['x1'] > bbox2['x2']:
        return bbox1['x1'] - bbox2['x2']
    elif bbox1['y2'] < bbox2['y1']:
        return bbox2['y1'] - bbox1['y2']
    elif bbox1['y1'] > bbox2['y2']:
        return bbox1['y1'] - bbox2['y2']
    elif bbox_is_completely_inside_the_other(bbox1, bbox2):
        return 0
    return 0.5


def bbox_is_completely_inside_the_other(bbox1: Bbox, bbox2: Bbox) -> bool:
    """Return True is one of the input bbox is completely inside the other."""
    if area_of_bbox(bbox1) >= area_of_bbox(bbox2):
        small_bbox = bbox2
        large_bbox = bbox1
    else:
        small_bbox = bbox1
        large_bbox = bbox2
    return small_bbox['x1'] >= large_bbox['x1'] and small_bbox['x2'] <= large_bbox['x2'] and \
           small_bbox['y1'] >= large_bbox['y1'] and small_bbox['y2'] <= large_bbox['y2']


def merge_bboxes(bboxes: List[Bbox]) -> Bbox:
    """Find single bbox which overlaps all bboxes in the input bbox list."""
    return {
        'x1': min([bbox['x1'] for bbox in bboxes]),
        'y1': min([bbox['y1'] for bbox in bboxes]),
        'x2': max([bbox['x2'] for bbox in bboxes]),
        'y2': max([bbox['y2'] for bbox in bboxes])
    }


def bboxes_to_regions(bboxes: List[Bbox], img: np.ndarray) -> List[np.ndarray]:
    """Takes a list of bounding boxes and returns a list of their corresponding region crops"""
    return list(map(lambda bbox: util.crop_image_region(img, bbox), bboxes))


def discard_regions_with_little_text(region_bbox_pairs: List[Tuple[np.ndarray, Bbox]],
                                     discard_text_percentage_cutoff: float,
                                     discard_area_percentage_cutoff: float,
                                     discard_area_absolute_cutoff: float) -> List[Tuple[np.ndarray, Bbox]]:
    """
    Discards regions which have less than a certain percentage of white pixels,
    and have an area that is less than a certain percentage of the max bbox area.
    These percentages are determined by the input parameters.
    """
    if not region_bbox_pairs:
        return region_bbox_pairs
    nintieth_percentile_area = np.percentile([area_of_bbox(rb_pair[1]) for rb_pair in region_bbox_pairs], 90)
    text_condition = lambda region: np.sum(region) / (255 * region.size) > discard_text_percentage_cutoff
    area_condition = lambda bbox: area_of_bbox(bbox) / nintieth_percentile_area > discard_area_percentage_cutoff or \
                                  area_of_bbox(bbox) > discard_area_absolute_cutoff
    filter_condition = lambda region_bbox_pair: text_condition(region_bbox_pair[0]) and \
                                                area_condition(region_bbox_pair[1])
    return list(filter(filter_condition, region_bbox_pairs))


def discard_bboxes_with_bad_aspect_ratio(bboxes: List[Bbox],
                                         aspect_ratio_percentage_cutoff: float,
                                         min_height_cutoff: float) -> List[Bbox]:
    """
    Discards bboxes with an aspect ratio that is too vertical, as determined by aspect_ratio_percentage_cutoff
    Also discards bboxes with a height less than min_height_cutoff
    """
    aspect_ratio_filter = lambda bbox: (bbox['y2'] - bbox['y1']) / (bbox['x2'] - bbox['x1']) < \
                                       aspect_ratio_percentage_cutoff
    min_height_filter = lambda bbox: bbox['y2'] - bbox['y1'] > min_height_cutoff
    return list(filter(lambda bbox: aspect_ratio_filter(bbox) and min_height_filter(bbox), bboxes))


def sort_bboxes(bboxes: List[Bbox]) -> List[Bbox]:
    """
    Takes a list of bboxes and returns them in raster scan order. It takes the output of
    decompose_into_lines, sorts the lines by y1 value of the max area bbox, and sorts each line
    by the x1 value of the bboxes
    """
    lines = decompose_into_lines(bboxes)
    lines.sort(key=lambda line: line[0]['y1'])
    for line in lines:
        line.sort(key=lambda bbox: bbox['x1'])
    return [bbox for line in lines for bbox in line]


def decompose_into_lines(bboxes: List[Bbox]) -> List[List[Bbox]]:
    """
    Takes a list of bboxes and returns a list of lines, where each line is a list of bboxes
    It does this by finding the max area bbox, adding all bboxes with a centroid within its y-range
    into one line, and recursing on the remaining bboxes
    """
    if not bboxes:
        return []
    max_area_bbox = max(bboxes, key=area_of_bbox)

    # this is equivalent to saying we want at least 50% of the bbox within the y region of the largest box
    y_overlap_condition = lambda bbox: centroid(bbox)['y'] > max_area_bbox['y1'] and \
        centroid(bbox)['y'] < max_area_bbox['y2']
    # this is equivalent to saying we want at least 50% of the bbox outside the x region of the largest box
    x_overlap_condition = lambda bbox: centroid(bbox)['x'] < max_area_bbox['x1'] or \
        centroid(bbox)['x'] > max_area_bbox['x2']

    current_line = list(filter(lambda bbox: y_overlap_condition(bbox) and x_overlap_condition(bbox), bboxes))
    current_line.append(max_area_bbox)

    remaining_bboxes = [bbox for bbox in bboxes if bbox not in current_line]
    return [current_line] + decompose_into_lines(remaining_bboxes)


def centroid(bbox):
    """Takes a bbox and returns its centroid"""
    return {
        'x': (bbox['x1'] + bbox['x2']) / 2,
        'y': (bbox['y1'] + bbox['y2']) / 2,
    }
