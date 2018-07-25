from time import time
from typing import Callable, Optional, Union, Tuple

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
import wandb
from wandb.keras import WandbCallback

from text_recognizer.train.gpu_util_sampler import GPUUtilizationSampler


EARLY_STOPPING = True
TENSORBOARD = False
GPU_UTIL_SAMPLER = False


def train_model(
        model: Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        loss: Union[str, Callable],
        gpu_ind: Optional[int]=None,
        use_wandb=False):
    model.compile(loss=loss, optimizer=RMSprop(), metrics=['accuracy'])

    callbacks = []

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)

    if GPU_UTIL_SAMPLER:
        gpu_utilization = GPUUtilizationSampler(gpu_ind)
        callbacks.append(gpu_utilization)

    if TENSORBOARD:
        tensorboard = TensorBoard(log_dir=f'logs/{time()}_{gpu_ind}')
        callbacks.append(tensorboard)

    if use_wandb:
        wandb = WandbCallback()
        callbacks.append(wandb)

    t = time()
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.25,
                        callbacks=callbacks,
                        verbose=1)
    print('Training took {:2f} s'.format(time() - t))

    if GPU_UTIL_SAMPLER:
        gpu_utilizations = gpu_utilization.samples
        print(f'GPU utilization: {round(np.mean(gpu_utilizations), 2)} +- {round(np.std(gpu_utilizations), 2)}')

    return history


def evaluate_model(model: Model, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    score = model.evaluate(x, y, verbose=1)
    print('Test loss/accuracy:', score[0], score[1])
    return score
