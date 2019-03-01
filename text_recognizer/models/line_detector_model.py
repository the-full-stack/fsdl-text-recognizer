"""Define LineModel class."""
from typing import Callable, Dict, Tuple

import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from text_recognizer.datasets.iam_paragraphs import IamParagraphsDataset
from text_recognizer.datasets.sequence import DatasetSequence
from text_recognizer.models.base import Model
from text_recognizer.networks import fcn


_DATASET_ARGS = {
    'image_shape': (256, 256)
}

_DATA_AUGMENTATION_PARAMS = {
    'width_shift_range': 0.06,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'zoom_range': 0.1,
    'fill_mode': 'constant',
    'cval': 0,
    'shear_range': 3,
}


class LineDetectorModel(Model):
    """Model for regions lines of ine of text."""
    def __init__(self,
                 dataset_cls: type = IamParagraphsDataset,
                 network_fn: Callable = fcn,
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values for this model."""
        if dataset_args is None:
            dataset_args = _DATASET_ARGS
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

        self.image_shape = dataset_args['image_shape']
        self.data_augmentor = ImageDataGenerator(**_DATA_AUGMENTATION_PARAMS)
        self.batch_augment_fn = self.augment_batch  # just rename augment_batch() to batch_augment_fn()

    def loss(self):  # pylint: disable=no-self-use
        return 'categorical_crossentropy'

    def optimizer(self):  # pylint: disable=no-self-use
        return Adam(0.001/2)

    def metrics(self):  # pylint: disable=no-self-use
        return None

    def fit(self, dataset, batch_size=32, epochs=1000, callbacks=None):
        """Overwriting the BaseModel fit method because we have to use ImageDataGenerator from keras."""
        if callbacks is None:
            callbacks = {}

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())

        train_sequence = DatasetSequence(
            dataset.x_train,
            dataset.y_train,
            batch_size,
            augment_fn=self.batch_augment_fn,
            format_fn=self.batch_format_fn
        )
        test_sequence = DatasetSequence(
            dataset.x_test,
            dataset.y_test,
            batch_size,
            augment_fn=None,
            format_fn=self.batch_format_fn
        )

        self.network.fit_generator(
            train_sequence,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test_sequence,  # (dataset.x_test, dataset.y_test),
            use_multiprocessing=True,
            workers=2,
            shuffle=True
        )

    def augment_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs different random transformations on the whole batch of x, y samples."""
        x_augment, y_augment = zip(*[self._augment_sample(x, y) for x, y in zip(x_batch, y_batch)])
        return np.stack(x_augment, axis=0), np.stack(y_augment, axis=0)

    def _augment_sample(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the same random image transformation on both x and y.
        x is a 2d image of shape self.image_shape, but self.data_augmentor needs the channel image too.
        """
        x_3d = np.expand_dims(x, axis=-1)
        transform_parameters = self.data_augmentor.get_random_transform(x_3d.shape)
        x_augment = self.data_augmentor.apply_transform(x_3d, transform_parameters)
        y_augment = self.data_augmentor.apply_transform(y, transform_parameters)
        return np.squeeze(x_augment, axis=-1), y_augment

    def predict_on_image(self, x: np.ndarray) -> np.ndarray:
        """Returns the network predictions on x."""
        return self.network.predict(np.expand_dims(x, axis=0))[0]

    def evaluate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 batch_size: int = 32,
                 verbose: bool = False) -> float:  # pylint: disable=unused-argument
        """Evaluates the network on x, y on returns the loss."""
        return self.network.evaluate(x, y, batch_size=batch_size)
