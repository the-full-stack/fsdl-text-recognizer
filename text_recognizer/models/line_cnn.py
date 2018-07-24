import pathlib
from typing import Tuple

from boltons.cacheutils import cachedproperty
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed, Lambda
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.model import Model
from text_recognizer.datasets.emnist_lines import EmnistLinesDataset
from text_recognizer.networks.cnn import lenet


DIRNAME = pathlib.Path(__file__).parents[0].resolve()
MODEL_NAME = pathlib.Path(__file__).stem
MODEL_WEIGHTS_FILENAME = DIRNAME / f'{MODEL_NAME}_weights.h5'


class LineCnn(Model):
    def __init__(self):
        np.random.seed(42)
        tf.set_random_seed(42)

        dataset = EmnistLinesDataset()
        self.mapping = dataset.mapping
        self.num_classes = len(self.mapping)
        self.max_length = dataset.max_length
        self.input_shape = dataset.input_shape

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        pred_raw = self.model.predict(np.expand_dims(image, 0), batch_size=1).squeeze()
        pred = convert_pred_raw_to_string(pred_raw, self.mapping)
        conf = np.min(np.max(pred_raw, axis=-1)) # The least confident of the predictions.
        return pred, conf

    loss = 'categorical_crossentropy'


class LineCnnFixedWidth(LineCnn):
    @cachedproperty
    def model(self):
        return create_fixed_width_image_model(self.input_shape, self.max_length, self.num_classes)


class LineCnnSlidingWindow(LineCnn):
    @cachedproperty
    def model(self):
        return create_sliding_window_image_model(self.input_shape, self.max_length, self.num_classes)


def convert_pred_raw_to_string(preds, mapping):
    return ''.join(mapping[label] for label in np.argmax(preds, axis=-1).flatten()).strip()


def create_fixed_width_image_model(image_shape: Tuple[int, int], max_length: int, num_classes: int) -> KerasModel:
    image_height, image_width = image_shape
    # output_shape = (num_classes, max_length)

    image_input = Input(shape=(image_height, image_width))

    window_width = image_width // max_length
    image_patches = Reshape((image_height, max_length, window_width, 1))(image_input)
    image_patches_permuted = Permute((2, 1, 3, 4))(image_patches)

    convnet = lenet(image_height, window_width, num_classes)
    convnet_outputs = TimeDistributed(convnet)(image_patches_permuted)

    model = KerasModel(inputs=image_input, outputs=convnet_outputs)
    model.summary()
    return model


def create_sliding_window_image_model(
        image_shape: Tuple[int, int],
        max_length: int,
        num_classes: int,
        window_width: int=12,
        window_stride: int=9) -> KerasModel:
    image_height, image_width = image_shape
    # output_shape = (num_classes, max_length)

    image_input = Input(shape=image_shape) # (image_height, image_width)
    window_width = 12
    window_stride = 8

    def slide_window(image, window_width=window_width, window_stride=window_stride):
        # import tensorflow as tf  # Might need this import for Keras to save/load properly
        # batch_size, image_height, image_width, num_channels = image.shape
        kernel = [1, 1, window_width, 1]
        strides = [1, 1, window_stride, 1]
        patches = tf.extract_image_patches(image, kernel, strides, [1, 1, 1, 1], 'SAME')
        patches = tf.transpose(patches, (0, 2, 1, 3))
        patches = tf.expand_dims(patches, -1)
        return patches

    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    image_patches = Lambda(slide_window)(image_reshaped)  # (num_windows, num_windows, window_width, 1)
    convnet = lenet(image_height, window_width)
    convnet_outputs = TimeDistributed(convnet)(image_patches)  # (num_windows, 128)
    convnet_outputs_flat = Flatten()(convnet_outputs) # (num_windows * 128)

    fc = Dense(max_length * 128, activation='relu')(convnet_outputs_flat) # (max_length * 128)
    fc_reshaped = Reshape((max_length, 128))(fc) # (max_length, 128)

    output = TimeDistributed(Dense(num_classes, activation='softmax'))(fc_reshaped) # (max_length, num_classes)
    model = KerasModel(inputs=image_input, outputs=output)
    model.summary()
    return model
