import pathlib
from typing import Tuple

from boltons.cacheutils import cachedproperty
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed, Lambda, ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window


def line_cnn_sliding_window(
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        window_width: float=16,
        window_stride: float=10) -> KerasModel:
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

    image_patches = Lambda(
        slide_window,
        arguments={'window_width': window_width, 'window_stride': window_stride}
    )(image_reshaped)
    # (num_windows, image_height, window_width, 1)

    # Make a LeNet and get rid of the last two layers (softmax and dropout)
    convnet = lenet((image_height, window_width, 1), (num_classes,))
    convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)

    convnet_outputs = TimeDistributed(convnet)(image_patches)
    # (num_windows, 128)

    # Now we have to get to (output_length, num_classes) shape. One way to do it is to do another sliding window with
    # width = floor(num_windows / output_length)
    # Note that this will likely produce too many items in the output sequence, so take only output_length,
    # and watch out that width is at least 2 (else we will only be able to predict on the first half of the line)

    ##### Your code below (Lab 2)
    convnet_outputs_extra_dim = Lambda(lambda x: tf.expand_dims(x, -1))(convnet_outputs)
    # (num_windows, 128, 1)

    num_windows = int((image_width - window_width) / window_stride) + 1
    width = int(num_windows / output_length)

    conved_convnet_outputs = Conv2D(num_classes, (width, 128), (width, 1), activation='softmax')(convnet_outputs_extra_dim)
    # (image_width / width, 1, num_classes)

    squeezed_conved_convnet_outputs = Lambda(lambda x: tf.squeeze(x, 2))(conved_convnet_outputs)
    # (max_length, num_classes)

    # Since we floor'd the calculation of width, we might have too many items in the sequence. Take only output_length.
    softmax_output = Lambda(lambda x: x[:, :output_length, :])(squeezed_conved_convnet_outputs)
    ##### Your code above (Lab 2)

    model = KerasModel(inputs=image_input, outputs=softmax_output)
    model.summary()
    return model
