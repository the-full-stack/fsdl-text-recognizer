# Lab 2

Add code to `text_recognizer/models/line_model.py` and `text_recognizer/networks/line_cnn_sliding_window.py` to make the following command train successfully.

```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistLinesDataset", "model": "LineModel", "network": "line_cnn_sliding_window"}'
```
