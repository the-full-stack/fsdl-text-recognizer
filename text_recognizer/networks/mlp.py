from typing import Tuple

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten


def mlp(num_classes: int, input_shape: Tuple[int, ...], layer_size: int=128, dropout_amount: float=0.2) -> Model:
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dropout(dropout_amount))
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dropout(dropout_amount))
    model.add(Dense(num_classes, activation='softmax'))
    return model
