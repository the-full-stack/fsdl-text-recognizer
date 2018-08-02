from typing import Callable, Dict, Tuple

import editdistance
import numpy as np
import tensorflow as tf

from text_recognizer.datasets.emnist_lines import EmnistLinesDataset
from text_recognizer.datasets.sequence import DatasetSequence
from text_recognizer.models.base import Model
from text_recognizer.networks import line_cnn_sliding_window


def loss_ignoring_blanks(target, output):
    """This is categorical crossentropy, but with targets that correspond to the padding symbol not counting."""
    import tensorflow.keras.backend as K
    output /= tf.reduce_sum(output, -1, True)
    _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    prod = target * tf.log(output)
    # TODO ??
    loss = - tf.reduce_sum(prod, -1)
    return loss


class LineModel(Model):
    def __init__(self, dataset_cls: type=EmnistLinesDataset, network_fn: Callable=line_cnn_sliding_window, dataset_args: Dict=None, network_args: Dict=None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def evaluate(self, x, y, verbose=True):
        ##### Your code should produce pred_strings and true_strings
        ##### Your code below (Lab 3)
        sequence = DatasetSequence(x, y)
        preds_raw = self.network.predict_generator(sequence)
        trues = np.argmax(y, -1)
        preds = np.argmax(preds_raw, -1)
        ## Stripping of the 3 characters. How does IAM lines deal with that?
        pred_strings = [''.join(self.data.mapping.get(label, '') for label in pred).strip(' |_') for pred in preds]
        true_strings = [''.join(self.data.mapping.get(label, '') for label in true).strip(' |_') for true in trues]
        ##### Your code above (Lab 3)
        char_accuracies = [
            1 - editdistance.eval(true_string, pred_string) / len(true_string)
            for pred_string, true_string in zip(pred_strings, true_strings)
        ]
        if verbose:
            sorted_ind = np.argsort(char_accuracies)
            print("\nLeast accurate predictions:")
            for ind in sorted_ind[:5]:
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')
            print("\nMost accurate predictions:")
            for ind in sorted_ind[-5:]:
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')
            print("\nRandom predictions:")
            for ind in np.random.randint(0, len(char_accuracies), 5):
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')
        mean_accuracy = np.mean(char_accuracies)
        return mean_accuracy

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).squeeze()
        pred = ''.join(self.data.mapping[label] for label in np.argmax(pred_raw, axis=-1).flatten()).strip()
        conf = np.min(np.max(pred_raw, axis=-1)) # The least confident of the predictions.
        return pred, conf

    # def loss(self):
    #     return loss_ignoring_blanks
