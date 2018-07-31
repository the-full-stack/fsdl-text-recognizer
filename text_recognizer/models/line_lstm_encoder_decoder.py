from boltons.cacheutils import cachedproperty
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, RepeatVector, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.cnn import lenet, slide_window


class LineLstmEncoderDecoder(LineModel):
    def __init__(self, window_width: int=14, window_stride: int=14):
        super().__init__()
        self.window_width = window_width
        self.window_stride = window_stride

    @cachedproperty
    def model(self):
        return create_sliding_window_rnn_model(self.input_shape, self.max_length, self.num_classes, self.window_width, self.window_stride)


def create_sliding_window_rnn_model(input_shape, max_length, num_classes, window_width, window_stride):
    def slide_window_bound(image, window_width=window_width, window_stride=window_stride):
        return slide_window(image, window_width, window_stride)

    encoder_dim = num_classes * max_length
    decoder_dim = num_classes

    image_height, image_width = input_shape
    image_input = Input(shape=input_shape)
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    image_patches = Lambda(slide_window_bound)(image_reshaped)  # (num_windows, image_height, window_width, 1)
    convnet = lenet(image_height, window_width)  # (image_height, window_width, 1) -> (128,)
    convnet_outputs = TimeDistributed(convnet)(image_patches)  # (num_windows, 128)

    gpu_present = len(device_lib.list_local_devices()) > 1
    lstm = CuDNNLSTM if gpu_present else LSTM

    # Your code below here (Lab 3)
    encoder_output = lstm(encoder_dim, return_sequences=False)(convnet_outputs) # (lstm_dim)
    repeated_encoding = RepeatVector(max_length)(encoder_output) # (max_length, lstm_dim)
    decoder_output = lstm(decoder_dim, return_sequences=True)(repeated_encoding)
    # Your code above here (Lab 3)

    softmax_outputs = TimeDistributed(Dense(num_classes, activation='softmax'))(decoder_output) # (max_length, num_classes)
    model = KerasModel(inputs=image_input, outputs=softmax_outputs)
    model.summary()
    return model
