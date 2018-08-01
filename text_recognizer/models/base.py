import pathlib

from boltons.cacheutils import cachedproperty
import inflection
import numpy as np
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import RMSprop

from text_recognizer.datasets.base import Dataset
from text_recognizer.datasets.sequence import DatasetSequence


DIRNAME = pathlib.Path(__file__).parents[0].resolve()


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""
    def weights_filename(self):
        model_name = inflection.underscore(self.__class__.__name__)
        return str(DIRNAME / f'{model_name}_weights.h5')

    def network(self) -> KerasModel:
        raise NotImplementedError

    def fit(self, dataset, batch_size=32, epochs=10, callbacks=[]):
        self.network.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        train_sequence = DatasetSequence(dataset.x_train, dataset.y_train, batch_size)
        test_sequence = DatasetSequence(dataset.x_test, dataset.y_test, batch_size)

        self.network.fit_generator(
            train_sequence,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test_sequence,
            use_multiprocessing=True,
            workers=1,
            shuffle=True
        )

    def evaluate(self, x, y):
        sequence = DatasetSequence(x, y)
        preds = self.network.predict_generator(sequence)
        # TODO: use evaluate_generator after getting rid of CTC
        return np.mean(np.argmax(preds, -1) == np.argmax(y, -1))

    @property
    def loss(self):
        return 'categorical_crossentropy'

    @cachedproperty
    def optimizer(self):
        return RMSprop()

    @property
    def metrics(self):
        return ['accuracy']

    def load_weights(self):
        self.network.load_weights(self.weights_filename())

    def save_weights(self):
        self.network.save_weights(self.weights_filename())
