# Lab 1 Instructions


## Training the network

You will have to add a tiny bit of code to `text_recognizer/networks/mlp.py` before being able to train.
When you finish writing your code, you can train a canonical model and save the weights. It will take about 5 minutes to download the dataset and train your model:

```sh
pipenv run training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp"}'
```

You can also run the above command as a shortcut: `tasks/train_character_predictor.py`

Just for fun, you can try a larger MLP, with a larger batch size

```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 8}} "train_args": {"batch_size": 256}'
```

## Testing

Your network is trained, but you need to write a bit more code to get the `CharacterModel` to use it to predict.
Open up `text_recognizer/models/character_model.py` and write some code there to make it work.
You can test that it works by running

```sh
pipenv run pytest -s text_recognizer/tests/test_character_predictor.py
```

Or, use the shorthand `tasks/run_prediction_tests.sh`

Testing should finish quickly.

## Submitting to Gradescope

Before submitting to Gradescope, commit and push your changes:

```sh
git commit -am "my work"
git push mine master
```

Now go to https://gradescope.com/courses/21098/assignments and click on Lab 1.
Select the Github submission option, and there select your fork of the `fsdl-text-recognizer-project` repo and the master branch, and click Submit.
Don't forget to enter a name for the leaderboard :)

The autograder treats code that is in `lab1/text_recognizer` as your submission, so make sure your code is indeed there.

The autograder should finish in <1 min, and display the results.
Your name will show up in the Leaderboard.
