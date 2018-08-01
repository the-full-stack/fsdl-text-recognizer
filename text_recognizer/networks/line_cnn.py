import pathlib
from typing import Tuple

from boltons.cacheutils import cachedproperty
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed, Lambda, ZeroPadding2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window


def line_cnn_sliding_window(
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        window_width: float=16,
        window_stride: float=8) -> KerasModel:
    """
    Input is an image with shape (image_height, image_width)
    Output is of shape (output_length, num_classes)
    """
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    image_input = Input(shape=input_shape)
    # (image_height, image_width)

    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    # (image_height, image_width, 1)

    # Conv block
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(image_reshaped)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # (image_height // 2 - 2, image_width // 2 - 2)

    # Conv block
    conv3 = Conv2D(32, kernel_size=(3, 3), activation='relu')(pool1)
    conv4 = Conv2D(64, (3, 3), activation='relu')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #



    convnet_outputs_extra_dim = Lambda(lambda x: tf.expand_dims(x, -1))(convnet_outputs)
    # (num_windows, 128, 1)

    # Now we have to get to (output_length, num_classes) shape. One way to do it is to do another sliding window with
    # width = floor(num_windows / output_length)
    # Note that it will likely produce too many items in the output sequence, so take only output_length.

    # Your code below here (Lab 2)
    num_windows = int((image_width - window_width) / window_stride) + 2
    width = int(num_windows / output_length)

    conved_convnet_outputs = Conv2D(num_classes, (width, 128), (width, 1), activation='softmax')(convnet_outputs_extra_dim)
    # (image_width / width, 1, num_classes)

    squeezed_conved_convnet_outputs = Lambda(lambda x: tf.squeeze(x, 2))(conved_convnet_outputs)
    # (max_length, num_classes)

    # Since we floor'd the calculation of width, we might have too many items in the sequence. Take only output_length.
    softmax_output = Lambda(lambda x: x[:, :output_length, :])(squeezed_conved_convnet_outputs)
    # Your code above here (Lab 2)

    model = KerasModel(inputs=image_input, outputs=softmax_output)
    return model
