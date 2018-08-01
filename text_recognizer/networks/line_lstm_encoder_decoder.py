from boltons.cacheutils import cachedproperty
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, RepeatVector, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window


def line_lstm_encoder_decoder_sw(input_shape, output_shape, window_width, window_stride):
    # Here is another way to pass arguments to the Keras Lambda function
    def slide_window_bound(image, window_width=window_width, window_stride=window_stride):
        return slide_window(image, window_width, window_stride)

    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    encoder_dim = num_classes * output_length // 8
    decoder_dim = num_classes

    image_input = Input(shape=input_shape)
    # (image_height, image_width)

    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    # (image_height, image_width, 1)

    image_patches = Lambda(slide_window_bound)(image_reshaped)
    # (num_windows, image_height, window_width, 1)

    convnet = lenet((image_height, window_width, 1), (num_classes,))
    convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
    # (image_height, window_width, 1) -> (128,)

    convnet_outputs = TimeDistributed(convnet)(image_patches)
    # (num_windows, 128)

    gpu_present = len(device_lib.list_local_devices()) > 1
    lstm = CuDNNLSTM if gpu_present else LSTM

    # Your code below (Lab 3)
    encoder_output = lstm(encoder_dim, return_sequences=False)(convnet_outputs)
    # (encoder_dim)
    repeated_encoding = RepeatVector(output_length)(encoder_output)
    # (max_length, encoder_dim)
    decoder_output = lstm(decoder_dim, return_sequences=True)(repeated_encoding)
    # (output_length, decoder_dim)
    # Your code above (Lab 3)

    softmax_output = TimeDistributed(Dense(num_classes, activation='softmax'))(decoder_output)
    # (max_length, num_classes)

    model = KerasModel(inputs=image_input, outputs=softmax_output)
    return model
