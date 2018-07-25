from boltons.cacheutils import cachedproperty
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.cnn import lenet


class LineLstmWithCtc:
    pass
    # TODO
    


class LineLstm(LineModel):
    @cachedproperty
    def model(self):
        return create_sliding_window_rnn_model(self.input_shape, self.max_length, self.num_classes, 28, 28)


def create_sliding_window_rnn_model(input_shape, max_length, num_classes, window_width, window_stride):
    def slide_window(image, window_width=window_width, window_stride=window_stride):
        kernel = [1, 1, window_width, 1]
        strides = [1, 1, window_stride, 1]
        patches = tf.extract_image_patches(image, kernel, strides, [1, 1, 1, 1], 'SAME')
        patches = tf.transpose(patches, (0, 2, 1, 3))
        patches = tf.expand_dims(patches, -1)
        return patches
    
    image_height, image_width = input_shape    
    image_input = Input(shape=input_shape)
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    image_patches = Lambda(slide_window)(image_reshaped)  # (num_windows, image_height, window_width, 1)
    convnet = lenet(image_height, window_width)
    convnet_outputs = TimeDistributed(convnet)(image_patches)  # (num_windows, 128)
    
    # LSTM outputting a single vector
    rnn_output = CuDNNLSTM(num_classes * max_length, return_sequences=False, go_backwards=True)(convnet_outputs) # (num_classes * max_length)
    reshaped_rnn_output = Reshape((max_length, num_classes))(rnn_output)
    softmaxed_outputs = TimeDistributed(Dense(num_classes, activation='softmax'))(reshaped_rnn_output)
    
    model = KerasModel(inputs=image_input, outputs=softmaxed_outputs)
    model.summary()
    return model
