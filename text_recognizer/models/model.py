import pathlib

import inflection


DIRNAME = pathlib.Path(__file__).parents[0].resolve()


class Model:
    def stem(self):
        return pathlib.Path(__file__).stem

    def weights_filename(self):
        model_name = inflection.underscore(self.__class__.__name__)
        return str(DIRNAME / f'{model_name}_weights.h5')

    def load_weights(self):
        self.model.load_weights(self.weights_filename())

    def save_weights(self):
        self.model.save_weights(self.weights_filename())
