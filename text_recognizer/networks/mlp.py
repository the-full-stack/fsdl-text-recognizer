from typing import Tuple

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten


def mlp(input_shape: Tuple[int, ...],
        num_classes: int,
        layer_size: int=128,
        dropout_amount: float=0.2,
        num_layers: int=2) -> Model:
    """Simple multi-layer perceptron."""
    # Your code below (Lab 1)
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    for _ in range(num_layers):
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout_amount))
    model.add(Dense(num_classes, activation='softmax'))
    # Your code above (Lab 1)
    return model
