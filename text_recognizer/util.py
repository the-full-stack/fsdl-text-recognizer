import os
from urllib.request import urlopen

import cv2


def read_image(image_uri, grayscale=False):
    """
    Read image_uri.
    """
    def read_image_from_filename(image_filename, grayscale):
        imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        return cv2.imread(image_filename, imread_flag)

    def read_image_from_url(image_url, grayscale):
        url_response = urlopen(image_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        return cv2.imdecode(img_array, imread_flag)

    local_file = os.path.exists(image_uri)
    try:
        img = None
        if local_file:
            img = read_image_from_filename(image_uri, grayscale)
        else:
            img = read_image_from_url(image_uri, grayscale)
    except Exception as e:
        raise ValueError("Could not load image at {}: {}".format(image_uri, e))
    return img


def write_image(image, filename):
    cv2.imwrite(filename, image)
