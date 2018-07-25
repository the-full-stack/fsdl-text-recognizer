import numpy as np
from tensorflow.keras.utils import Sequence


class CtcDatasetSequence(Sequence):
    """
    Implementation of https://keras.io/utils/#sequence for CTC models,
    which need to compute the loss in the network.
    """
    def __init__(self, x, y, batch_size, output_sequence_length):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.output_sequence_length = output_sequence_length

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = np.take(self.x, range(idx * self.batch_size, (idx + 1) * self.batch_size), axis=0, mode='clip')
        batch_y = np.take(self.y, range(idx * self.batch_size, (idx + 1) * self.batch_size), axis=0, mode='clip')
        y_true = np.argmax(batch_y, -1)
        batch_inputs = {
            'image': batch_x,
            'y_true': y_true,
            'input_length': np.ones((self.batch_size, 1)) * self.output_sequence_length,
            'label_length': np.array([np.where(batch_y[ind, :, -1] == 1)[0][0] for ind in range(self.batch_size)])
        }
        batch_outputs = {
            'ctc_loss': np.zeros(batch_y.shape[0]),  # dummy
            'ctc_decoded': y_true
        }
        return batch_inputs, batch_outputs
