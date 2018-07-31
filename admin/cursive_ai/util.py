import base64
import os
from urllib.request import urlopen

import numpy as np
import cv2


def read_image(image_uri, grayscale=False):
    """
    Read image_uri.
    """
    def read_image_from_filename(image_filename, imread_flag):
        return cv2.imread(image_filename, imread_flag)

    def read_image_from_url(image_url, imread_flag):
        url_response = urlopen(image_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, imread_flag)

    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    local_file = os.path.exists(image_uri)
    try:
        img = None
        if local_file:
            img = read_image_from_filename(image_uri, imread_flag)
        else:
            img = read_image_from_url(image_uri, imread_flag)
        assert img is not None
    except Exception as e:
        raise ValueError("Could not load image at {}: {}".format(image_uri, e))
    return img


def read_b64_image(b64_string, grayscale=False):
    """
    Loads base64-encoded images such as "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAB80lEQVQoFX3BTUiTcRzA8e/v2eNeQlsvBqbSehYkEuWhwHnqopRooRP10LWLEBRYiWW4SrqsW0V07mIQYmEQEb0g4QQvKTjI2ia4TXxrwtaKPfpvOjNXsM9HKEAoQPifJmtsEv4h1qojrnfziwoQ8lmM2qsHi79M3FhZByGPzdnY0GKzJIOdcyYIO4jdW9/oJCklkfOhDAg7OMp9nsPMBnfXxtqn0yBsK3I2NXitS1OPghd6zFtDERC2ldZcOuHKTLx9kuy+qQYGZ0DYIvaOliYtPO6PLut9vWpgcAaEHHF7fBVEfYGQsux/fO571+giCDn2trNt1nDAF0tjP/q0eq59Og3CJkvz/QqivkBIgbv1TlG4OWyCsMHi6L5mnx3rj6VBTnZeJtwcNkHI0k7XXTz00z88qcBRfrvOSAU75kwQsvSu1jqi3WMLIG6Pr8IWGr2+sg5CluOu1zU57E8rdpX3e9yy0PPpmwIE0CufV//oHQ2xt8xbc8Zmrg73La+RJYBuvHQvXvma0o4f85aVytLkg9e/2CCAbowYKpZRsqfYTmb1xcgrk00C6MaIoa8p0DRYmHr4OUKOAHrls+piRGFm4omh8Y8mWwTQ9t07VVWEMuOJ9/Nv4kv8IWTJgbL6EtZTHxLxjMlfQo5Olkk+oQChgN/vs7MdVwJwzAAAAABJRU5ErkJggg=="
    """
    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    try:
        _, b64_data = b64_string.split(',')
        return cv2.imdecode(np.frombuffer(base64.b64decode(b64_data), np.uint8), imread_flag)
    except Exception as e:
        raise ValueError("Could not load image from b64 {}: {}".format(b64_string, e))


def write_image(image, filename):
    cv2.imwrite(filename, image)


def resize(image, scale_factor):
    """
    Resize image.
    OpenCV does not resize in place, hence need to copy and return in case
    scale_factor is 1 to maintain consistency in behaviour
    """
    if scale_factor == 1:
        return image.copy()
    return cv2.resize(
        image,
        dsize=None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_AREA
    )


def binarize_image(img):
    """
    If img is not black and white, return binarized using Otsu's method.
    """
    if len(np.unique(img)) > 2:
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img


def crop_image_region(image, region=None):
    """
    Input
        image (2d ndarray of uint8)
        region (dict): must contain 'x1', 'x2', 'y1', 'y2' keys
                       note that 'units' key can be given, with '%' as a valid value
    Output
        image_region (2d ndarray of uint8)
    """
    if not region:
        return image
    H, W = image.shape[:2]
    if 'units' in region and region['units'] == '%':
        y1 = int(max(0, float(region['y1'])) * H / 100.)
        y2 = int(min(100, float(region['y2'])) * H / 100.)
        x1 = int(max(0, float(region['x1'])) * W / 100.)
        x2 = int(min(100, float(region['x2'])) * W / 100.)
    else:
        y1 = max(0, int(region['y1']))
        y2 = min(H, int(region['y2']))
        x1 = max(0, int(region['x1']))
        x2 = min(W, int(region['x2']))
    return image[y1:y2, x1:x2]
