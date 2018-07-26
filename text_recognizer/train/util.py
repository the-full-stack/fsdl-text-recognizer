from time import time
from typing import Callable, Optional, Union, Tuple

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import RMSprop
import wandb
from wandb.keras import WandbCallback

from text_recognizer.datasets.base import Dataset
from text_recognizer.models.base import Model
from text_recognizer.train.gpu_util_sampler import GPUUtilizationSampler


EARLY_STOPPING = True
TENSORBOARD = False
GPU_UTIL_SAMPLER = False


def train_model(model: Model, dataset: Dataset, epochs: int, batch_size: int, gpu_ind: Optional[int]=None, use_wandb=False) -> Model:
    callbacks = []

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
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
    history = model.fit(dataset, batch_size, epochs, callbacks)
    print('Training took {:2f} s'.format(time() - t))

    if GPU_UTIL_SAMPLER:
        gpu_utilizations = gpu_utilization.samples
        print(f'GPU utilization: {round(np.mean(gpu_utilizations), 2)} +- {round(np.std(gpu_utilizations), 2)}')

    return model


def evaluate_model(model: Model, dataset: Dataset) -> float:
    metric = model.evaluate(dataset.x_test, dataset.y_test)
    print('Test metric:', metric)
    return metric
