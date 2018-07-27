from typing import Tuple

from boltons.cacheutils import cachedproperty
import editdistance
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.ctc_dataset_sequence import CtcDatasetSequence
from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.cnn import lenet
from text_recognizer.networks.ctc import ctc_decode


class LineLstmWithCtc(LineModel):
    def __init__(self, window_width: int=14, window_stride: int=14):
        super().__init__()
        self.window_width = window_width
        self.window_stride = window_stride
        # TODO: compute this in terms of window_width, window_stride, and self.input_shape
        self.max_sequence_length = self.max_length * 2

    @cachedproperty
    def model(self):
        return create_sliding_window_rnn_with_ctc_model(self.input_shape, self.max_length, self.num_classes, self.window_width, self.window_stride)

    def fit(self, dataset, batch_size, epochs, callbacks):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        train_sequence = CtcDatasetSequence(dataset.x_train, dataset.y_train, batch_size, self.max_sequence_length)
        test_sequence = CtcDatasetSequence(dataset.x_test, dataset.y_test, batch_size, self.max_sequence_length)

        self.model.fit_generator(
            train_sequence,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test_sequence,
            use_multiprocessing=True,
            workers=1,
            shuffle=True
        )

    def evaluate(self, x, y, batch_size: int=32) -> float:
        decoding_model = KerasModel(inputs=self.model.input, outputs=self.model.get_layer('ctc_decoded').output)
        test_sequence = CtcDatasetSequence(x, y, batch_size, self.max_sequence_length)
        preds = decoding_model.predict_generator(test_sequence)
        trues = np.argmax(y, -1)
        pred_strings = [''.join(self.mapping.get(label, '') for label in pred).strip() for pred in preds]
        true_strings = [''.join(self.mapping.get(label, '') for label in true).strip() for true in trues]
        char_accuracies = [
            1 - editdistance.eval(true_string, pred_string) / len(pred_string)
            for true_string, pred_string in zip(pred_strings, true_strings)
        ]
        return np.mean(char_accuracies)

    @property
    def loss(self):
        """Dummy loss function: just pass through the loss we compute in the network."""
        return {'ctc_loss': lambda y_true, y_pred: y_pred}

    @property
    def metrics(self):
        return None

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        softmax_output_fn = K.function(
            [self.model.get_layer('image').input, K.learning_phase()],
            [self.model.get_layer('softmax_output').output]
        )
        input_image = np.expand_dims(image, 0)
        softmax_output = softmax_output_fn([input_image, 0])[0]
        decoded, log_prob = K.ctc_decode(softmax_output, np.array([64])) # TODO: don't hardcode 64: compute based on image and properties of self
        pred_raw = K.eval(decoded[0])[0]
        pred = ''.join(self.mapping[label] for label in pred_raw).strip()
        # conf = K.eval(K.softmax(log_prob))[0][0]
        neg_sum_logit = K.eval(log_prob)[0][0]
        conf = np.exp(neg_sum_logit) / (1 + np.exp(neg_sum_logit))
        # TODO: not sure if conf calculation is correct
        return pred, conf


def create_sliding_window_rnn_with_ctc_model(input_shape, max_length, num_classes, window_width, window_stride):
    def slide_window(image, window_width=window_width, window_stride=window_stride):
        kernel = [1, 1, window_width, 1]
        strides = [1, 1, window_stride, 1]
        patches = tf.extract_image_patches(image, kernel, strides, [1, 1, 1, 1], 'SAME')
        patches = tf.transpose(patches, (0, 2, 1, 3))
        patches = tf.expand_dims(patches, -1)
        return patches

    image_height, image_width = input_shape
    image_input = Input(shape=input_shape, name='image')
    y_true = Input(shape=(max_length,), name='y_true')
    input_length = Input(shape=(1,), name='input_length')
    label_length = Input(shape=(1,), name='label_length')

    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    image_patches = Lambda(slide_window)(image_reshaped)  # (num_windows, image_height, window_width, 1)
    convnet = lenet(image_height, window_width)
    convnet_outputs = TimeDistributed(convnet)(image_patches)  # (num_windows, 128)

    if len(device_lib.list_local_devices()) > 1:
        rnn_output = CuDNNLSTM(128, return_sequences=True)(convnet_outputs) # (sequence_length, 128)
    else:
        rnn_output = LSTM(128, return_sequences=True)(convnet_outputs) # (sequence_length, 128)
    softmax_output = TimeDistributed(Dense(num_classes, activation='softmax'), name='softmax_output')(rnn_output) # (sequence_length, 128)

    ctc_loss_output = Lambda(
        lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]),
        name='ctc_loss'
    )([y_true, softmax_output, input_length, label_length])

    ctc_decoded_output = Lambda(
        lambda x: ctc_decode(x[0], x[1]),
        name='ctc_decoded'
    )([softmax_output, input_length])

    model = KerasModel(inputs=[image_input, y_true, input_length, label_length], outputs=[ctc_loss_output, ctc_decoded_output])
    model.summary()
    return model
