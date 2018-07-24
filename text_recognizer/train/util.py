from time import time
from typing import Callable, Optional, Union, Tuple

# import gpustat
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
# import wandb
# from wandb.keras import WandbCallback


class GPUUtilizationSampler(Callback):
    """
    Measure GPU utilization at the end of 1% of all batches.
    (The more frequent the measuring, the slower and less accurate this callback becomes.)
    """
    def __init__(self, gpu_ind):
        self.gpu_ind = gpu_ind
        super()

    def on_train_begin(self, logs={}):
        self.samples = []

    def on_batch_end(self, batch, logs={}):
        if np.random.rand() > 0.99:
            gpu_info = gpustat.GPUStatCollection.new_query()[self.gpu_ind]
            self.samples.append(gpu_info.utilization)


def evaluate_model(model: Model, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    score = model.evaluate(x, y, verbose=1)
    print('Test loss/accuracy:', score[0], score[1])
    return score


def train_model(
        model: Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        loss: Union[str, Callable],
        gpu_ind: Optional[int]=None):
    model.compile(loss=loss, optimizer=RMSprop(), metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=3, verbose=1, mode='auto')
    gpu_utilization = GPUUtilizationSampler(gpu_ind)
    # tensorboard = TensorBoard(log_dir=f'logs/{time()}_{gpu_ind}')
    # wandb = WandbCallback()
    # callbacks = [early_stopping, gpu_utilization, tensorboard, wandb]
    callbacks = [early_stopping]
    t = time()
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.25,
                        callbacks=callbacks,
                        verbose=1)
    print('Training took {:2f} s'.format(time() - t))
    # gpu_utilizations = gpu_utilization.samples
    # print(f'GPU utilization: {round(np.mean(gpu_utilizations), 2)} +- {round(np.std(gpu_utilizations), 2)}')
    return history
