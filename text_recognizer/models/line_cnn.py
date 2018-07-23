import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed
from tensorflow.keras.models import Model, Sequential

from text_recognizer.datasets.emnist_lines import EmnistLinesDataset
from text_recognizer.networks.cnn import lenet


class LineCnn:
    def __init__(self):
        np.random.seed(42)
        tf.set_random_seed(42)

        self.dataset = EmnistLinesDataset(num_train=10, num_test=2)
        sample_image = self.dataset.x_train[0]
        h, w = sample_image.shape
        self.model = create_fixed_width_image_model(h, w, self.dataset.max_length, self.dataset.num_classes)


def create_fixed_width_image_model(image_height: int, image_width: int, max_length: int, num_classes: int) -> Model:
    input_shape = (image_height, image_width)
    output_shape = (num_classes, max_length)

    image_input = Input(shape=input_shape)

    window_width = image_width // max_length
    image_patches = Reshape((image_height, max_length, window_width, 1))(image_input)
    image_patches_permuted = Permute((2, 1, 3, 4))(image_patches)

    convnet = lenet(image_height, window_width, num_classes)
    output = TimeDistributed(convnet)(image_patches_permuted)

    model = Model(inputs=image_input, outputs=output)
    model.summary()
    return model
