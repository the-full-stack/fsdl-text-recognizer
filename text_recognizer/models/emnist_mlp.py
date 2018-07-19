import pathlib

from tensorflow.python.keras.models import load_model, Model, Sequential
from tensorflow.python.keras.layers import Activation, Dense, Dropout


def create_mlp_model(num_classes: int, input_size: int, layer_size: int=128, dropout_amount: float=0.2) -> Model:
    model = Sequential()
    model.add(Dense(layer_size, activation='relu', input_shape=(input_size,)))
    model.add(Dropout(dropout_amount))
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dropout(dropout_amount))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model
