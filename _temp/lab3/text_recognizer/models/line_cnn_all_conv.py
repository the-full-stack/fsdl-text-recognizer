import pathlib
from typing import Tuple

from boltons.cacheutils import cachedproperty
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed, Lambda, ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.cnn import lenet


class LineCnnAllConv(LineModel):
    def __init__(self, window_fraction: float=0.5, window_stride: float=0.5):
        super().__init__()
        self.window_fraction = window_fraction
        self.window_stride = window_stride

    @cachedproperty
    def model(self):
        return create_all_conv_model(self.input_shape, self.max_length, self.num_classes, self.window_fraction, self.window_stride)


def create_all_conv_model(
        image_shape: Tuple[int, int],
        max_length: int,
        num_classes: int,
        window_width_fraction: float=0.5,
        window_stride_fraction: float=0.5) -> KerasModel:
    image_height, image_width = image_shape

    model = Sequential()
    model.add(Reshape((image_height, image_width, 1), input_shape=image_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    new_height = image_height // 2
    new_letter_width = image_width // max_length // 2
    window_width = int(new_letter_width * window_width_fraction)
    window_stride = int(new_letter_width * window_stride_fraction)
    model.add(Conv2D(128, (new_height, window_width), (1, window_stride), activation='relu'))
    model.add(ZeroPadding2D(padding=((0, 0), (0, 1))))
    model.add(Dropout(0.5))

    width = int(1 / window_stride_fraction)
    model.add(Conv2D(num_classes, (1, width), (1, width), activation='softmax'))
    model.add(Lambda(lambda x: tf.squeeze(x, 1)))
    model.summary()
    return model
