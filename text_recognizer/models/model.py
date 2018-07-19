import tensorflow.python.keras.models as keras_models

MODELS_DIRNAME = pathlib.Path(__file__).parents[2].resolve() / 'models' / 'emnist_mlp'
MODEL_FILENAME = MODELS_DIRNAME / 'model.h5'

def load_model(model_name, experiment_name):
    filename = MODELS_DIRNAME / model_name / f'{experiment_name}.h5'
    keras_models.load_model(filename)

