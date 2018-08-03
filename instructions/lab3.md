# Lab 3

In this lab we'll keep working with the EmnistLines dataset.

We will be implementing LSTM model with CTC loss.
CTC loss needs to be implemented kind of strangely in Keras: by passing in all required data to compute the loss as inputs to the network (including the true label).
This is an example of a multi-input / multi-output network.

The relevant files to review are `text_recognizer/models/line_model_ctc.py`, which shows the batch formatting that needs to happen for the CTC loss to be computed inside of the network.

## Train LSTM model with CTC loss

You need to write code in `text_recognizer/networks/line_lstm.py` to make training work.
Training can be done via

```sh
pipenv run python training/run_experiment.py --save '{"dataset": "EmnistLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
```

or the shortcut `tasks/train_line_predictor.sh`

## Make sure the model is able to predict

You will also need to write some code in `text_recognizer/models/line_model_ctc.py` to predict on images.
After that, you should see tests pass when you run

```sh
pipenv run pytest -s text_recognizer/tests/test_line_predictor.py
```

Or you can do `tasks/run_prediction_tests.sh`, which will also run the CharacterModel tests.

## Things to try

If you have time left over, or want to play around with this later on, you can try using the `line_lstm` network, defined in `text_recognizer/networks/line_lstm.py`.
Code up an encoder-decoder architecture, for example!
