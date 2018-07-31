from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def lenet(image_height: int, image_width: int, num_classes: Optional[int] = None, expand_dims: bool=False) -> Model:
    model = Sequential()
    if expand_dims:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=(image_height, image_width, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    if num_classes:
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
    return model


def lenet_all_conv(image_height: int, image_width: int, num_classes: Optional[int] = None) -> Model:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Your lab2 code below here
    model.add(Conv2D(128, (image_height // 2 - 2, image_width // 2 - 2), activation='relu'))
    if num_classes:
        model.add(Dropout(0.5))
        model.add(Conv2D(num_classes, (1, 1), activation='softmax'))
    model.add(Flatten())
    # Your lab2 code above here
    return model


def slide_window(image, window_width, window_stride):
    """
    Takes (image_height, image_width, 1) input,
    Returns (num_windows, image_height, window_width, 1) output, where
    num_windows is floor((image_width - window_width) / window_stride) + 1
    """
    kernel = [1, 1, window_width, 1]
    strides = [1, 1, window_stride, 1]
    patches = tf.extract_image_patches(image, kernel, strides, [1, 1, 1, 1], 'SAME')
    patches = tf.transpose(patches, (0, 2, 1, 3))
    patches = tf.expand_dims(patches, -1)
    return patches
