# Lab 2

## LeNet

Add enough code to be able to run
```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "lenet"}'
```

## Emnist Lines

## Train on EmnistLines

Add code to `text_recognizer/models/line_model.py` and `text_recognizer/networks/line_cnn_sliding_window.py` to make the following command train successfully.

```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistLinesDataset", "model": "LineModel", "network": "line_cnn_sliding_window"}'
```
