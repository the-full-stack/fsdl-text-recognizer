import tensorflow as tf
import tensorflow.keras.backend as K


def slide_window(image, image_height, window_width, window_stride):
    """
    Takes (image_height, image_width, 1) input,
    Returns (num_windows, image_height, window_width, 1) output, where
    num_windows is floor((image_width - window_width) / window_stride) + 1
    """
    height = 1
    kernel = [1, height, window_width, 1]
    strides = [1, height, window_stride, 1]
    patches = tf.extract_image_patches(image, kernel, strides, [1, 1, 1, 1], 'VALID')
    patches = tf.transpose(patches, (0, 2, 1, 3))

    batch_size = K.shape(patches)[0]
    num_unroll = K.shape(patches)[1]

    expected_shape = (batch_size, num_unroll, image_height, window_width, 1)
    patches = K.reshape(patches, shape=expected_shape)
    return patches
