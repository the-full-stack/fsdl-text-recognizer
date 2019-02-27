"""Module implementing a lot of data augmentation techniques."""
from skimage import exposure
import numpy as np


def noop(image: np.ndarray) -> np.ndarray:
    """Performs no operation on the image."""
    return image


def random_intensity_stretch(image: np.ndarray) -> np.ndarray:
    """
    The image intensities are stretched from (random_p_min, random_p_max) to (0, 255).
    Output is of the same dtype as input image.
    """
    return _random_intensity_rescale(image, 'stretch')


def random_intensity_shrink(image: np.ndarray) -> np.ndarray:
    """
    The image intensities are shrunk from (image.min(), image.max()) to (random_p_min, random_p_max).
    Output is of the same dtype as input image.
    """
    return _random_intensity_rescale(image, 'shrink')


def _random_intensity_rescale(image: np.ndarray, rescale_type: str) -> np.ndarray:
    """The image intensities are rescaled. Output is of the same dtype as input image."""
    if np.issubdtype(image.dtype, np.integer):
        int_image = image
    else:
        int_image = (image * 255).astype(int)

    p_min = min(50, np.percentile(int_image, 2))
    p_max = max(200, np.percentile(int_image, 98))
    random_p_min = np.random.randint(0, p_min + 1)
    random_p_max = np.random.randint(p_max, 256)

    if rescale_type == 'stretch':
        int_processed_image = exposure.rescale_intensity(int_image,
                                                         in_range=(random_p_min, random_p_max),
                                                         out_range=(0, 255))
    else:
        int_processed_image = exposure.rescale_intensity(int_image,
                                                         in_range='image',
                                                         out_range=(random_p_min, random_p_max))

    if np.issubdtype(image.dtype, np.integer):
        return int_processed_image
    return (int_processed_image / 255.).astype(image.dtype)


def random_gamma_correction(image: np.ndarray) -> np.ndarray:
    """
    For gamma > 1, image becomes darker, else image becomes brighter.
    Output is of the same dtype as input image.
    """
    return exposure.adjust_gamma(image, gamma=np.random.uniform(0.7, 1.5))


def sigmoid_correction(image: np.ndarray) -> np.ndarray:
    """
    Pushes image histogram towards 0 and 255.
    Output is of the same dtype as input image.
    """
    return exposure.adjust_sigmoid(image)


def random_logarithmic_correction(image: np.ndarray) -> np.ndarray:
    """
    Gives a gray background to image by capping the maximum intensity.
    Output is of the same dtype as input image.
    """
    return exposure.adjust_log(image, gain=np.random.uniform(0.7, 1))


def random_adaptive_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Performs local constrast enhancement (CLAHE) on the image.
    Randomness is induced by randomly setting the clip limits.
    Output is of the same dtype as input image.
    """
    if image.ndim == 3:
        img = np.squeeze(image, axis=-1)
    else:
        img = image
    equalized_float_image = exposure.equalize_adapthist(img, clip_limit=np.random.uniform(0.01, 0.03))
    if image.dtype == 'uint8':
        return (equalized_float_image * 255).astype('uint8').reshape(image.shape)
    return equalized_float_image.reshape(image.shape)
