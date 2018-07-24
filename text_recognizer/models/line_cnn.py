import pathlib
from typing import Tuple

from boltons.cacheutils import cachedproperty
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed, Lambda, ZeroPadding2D
from tensorflow.keras.models import Sequential
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
    def __init__(self, window_fraction: float=0.5, window_stride: float=0.5):
        super().__init__()
        self.window_fraction = window_fraction
        self.window_stride = window_stride

    @cachedproperty
    def model(self):
        return create_sliding_window_image_model(self.input_shape, self.max_length, self.num_classes, self.window_fraction, self.window_stride)


class LineCnnAllConv(LineCnn):
    def __init__(self, window_fraction: float=0.5, window_stride: float=0.5):
        super().__init__()
        self.window_fraction = window_fraction
        self.window_stride = window_stride

    @cachedproperty
    def model(self):
        return create_all_conv_model(self.input_shape, self.max_length, self.num_classes, self.window_fraction, self.window_stride)


def convert_pred_raw_to_string(preds, mapping):
    return ''.join(mapping[label] for label in np.argmax(preds, axis=-1).flatten()).strip()


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


def create_sliding_window_image_model(
        image_shape: Tuple[int, int],
        max_length: int,
        num_classes: int,
        window_width_fraction: float=0.5,
        window_stride_fraction: float=0.5) -> KerasModel:
    image_height, image_width = image_shape
    image_input = Input(shape=image_shape)

    letter_width = image_width // max_length
    window_width = int(letter_width * window_width_fraction)
    window_stride = int(letter_width * window_stride_fraction)
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

    convnet = lenet(image_height, window_width)  # Note that this doesn't include the top softmax layer
    convnet_outputs = TimeDistributed(convnet)(image_patches)  # (num_windows, 128)
    convnet_outputs_extra_dim = Lambda(lambda x: tf.expand_dims(x, -1))(convnet_outputs) # (num_windows, 128, 1)

    width = int(1 / window_stride_fraction)
    conved_convnet_outputs = Conv2D(num_classes, (width, 128), (width, 1), activation='softmax')(convnet_outputs_extra_dim) # (max_length, 1, num_classes)
    conved_convnet_outputs_squeezed = Lambda(lambda x: tf.squeeze(x, 2))(conved_convnet_outputs) # (max_length, num_classes)

    model = KerasModel(inputs=image_input, outputs=conved_convnet_outputs_squeezed)
    model.summary()
    return model


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
