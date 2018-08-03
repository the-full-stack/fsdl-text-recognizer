# Lab 2

## LeNet

Add enough code to `lenet.py` to be able to run

```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "lenet"}'
```

Training will take a few minutes to train.

You should see an improvement in accuracy compared to the MLP model we trained in the previous lab.

## Emnist Lines

(Explain generating the dataset, look at the notebook)

## Train a Sliding Window LeNet on EmnistLines

Add code to `text_recognizer/networks/line_cnn_sliding_window.py` to make the following command train successfully.

```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistLinesDataset", "model": "LineModel", "network": "line_cnn_sliding_window"}'
```


You should be able to get to ~70% character accuracy with the default params.

## Train an all-conv model on EmnistLines

The previous model works, but is inefficient, as it re-does convolutions from scratch in every window.
Because we are using a simple LeNet, we can convert the whole model into all convolutions.
Write code in `text_recognizer/networks/line_cnn_all_conv.py` to make this happen.

When you train with

```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistLinesDataset", "model": "LineModel", "network": "line_cnn_all_conv", "network_args": {"window_width": 16, "window_stride": 8}}'
```

You should be getting roughly the same accuracy, and see slightly shorter runtimes: for me, it's 77ms/step vs 100ms/step for the previous model.

The amount of speedup depends on exactly how much windows overlap, so play around with those parameters.
For example:

```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistLinesDataset", "model": "LineModel", "network": "line_cnn_all_conv", "network_args": {"window_width": 8, "window_stride": 4}}'
```
