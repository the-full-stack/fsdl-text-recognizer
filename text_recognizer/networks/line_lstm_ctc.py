from boltons.cacheutils import cachedproperty
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, RepeatVector, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window
from text_recognizer.networks.ctc import ctc_decode


def get_input_length_tensor(image_patches, num_windows):
    """Return input_length_tensor of shape (batch_size, 1) with values num_windows."""
    batch_size = K.shape(image_patches)[0]
    input_length_tensor = K.ones((batch_size, 1), dtype='float32') * num_windows
    return input_length_tensor


def line_lstm_ctc(input_shape, output_shape, window_width=28, window_stride=14):
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    image_input = Input(shape=(image_height, image_width), name='image')
    y_true = Input(shape=(output_length,), name='y_true')
    label_length = Input(shape=(1,), name='label_length')

    gpu_present = len(device_lib.list_local_devices()) > 1
    lstm_fn = CuDNNLSTM if gpu_present else LSTM

    # Your code should use slide_window and extract image patches from image_input.
    # Pass a convolutional model over each image patch to generate a feature vector per window.
    # Pass these features through one or more LSTM layers.
    # Convert the lstm outputs to softmax outputs.
    # Note that lstms expect a input of shape (num_batch_size, num_timesteps, feature_length).
    softmax_output = None

    ##### Your code below (Lab 3)
    image_reshaped = Lambda(lambda x: K.expand_dims(x, axis=-1))(image_input)
    # (image_height, image_width, 1)

    image_patches = Lambda(
        slide_window,
        arguments={'image_height': image_height, 'window_width': window_width, 'window_stride': window_stride},
        name='slide_window'
    )(image_reshaped)
    # (num_windows, image_height, window_width, 1)

    # Make a LeNet and get rid of the last two layers (softmax and dropout)
    convnet = lenet((image_height, window_width, 1), (num_classes,))
    convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
    convnet_outputs = TimeDistributed(convnet)(image_patches)
    # (num_windows, 128)

    lstm_output = lstm_fn(128, return_sequences=True)(convnet_outputs)
    # (num_windows, 128)

    softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)
    # (num_windows, num_classes)
    ##### Your code above (Lab 3)

    ### Computing ctc loss and decoding softmax output below
    num_windows = int((image_width - window_width) / window_stride) + 1

    input_length = Lambda(
        get_input_length_tensor,
        arguments={'num_windows': num_windows}
    )(image_patches)

    ctc_loss_output = Lambda(
        lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]),
        name='ctc_loss'
    )([y_true, softmax_output, input_length, label_length])

    ctc_decoded_output = Lambda(
        lambda x: ctc_decode(x[0], x[1], output_length),
        name='ctc_decoded'
    )([softmax_output, input_length])

    model = KerasModel(
        inputs=[image_input, y_true, label_length],
        outputs=[ctc_loss_output, ctc_decoded_output]
    )
    return model
