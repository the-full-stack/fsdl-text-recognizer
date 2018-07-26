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


class LineCnnFixedWidth(LineModel):
    @cachedproperty
    def model(self):
        return create_fixed_width_image_model(self.input_shape, self.max_length, self.num_classes)


def create_fixed_width_image_model(image_shape: Tuple[int, int], max_length: int, num_classes: int) -> KerasModel:
    image_height, image_width = image_shape

    image_input = Input(shape=image_shape)

    window_width = image_width // max_length
    image_patches = Reshape((image_height, max_length, window_width, 1))(image_input)
    image_patches_permuted = Permute((2, 1, 3, 4))(image_patches)

    convnet = lenet(image_height, window_width, num_classes)
    convnet_outputs = TimeDistributed(convnet)(image_patches_permuted)

    model = KerasModel(inputs=image_input, outputs=convnet_outputs)
    model.summary()
    return model
