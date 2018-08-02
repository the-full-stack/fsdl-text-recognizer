# Lab 1 Instructions

## Checking out the repo

Start by cloning the repo (you may already have it present in your home directory), then going into it and pulling the latest version.

```sh
git clone https://github.com/gradescope/fsdl-text-recognizer-project.git
cd fsdl-text-recognizer-project
git pull origin master
```

Now go to https://github.com/gradescope/fsdl-text-recognizer-project and click Fork in the top right.
Select your personal account to fork to, and note down your USERNAME, which will be right after https://github.com in the URL that you will be taken to.
Add your fork as a remote of the repo and push to it

```sh
git remote add mine https://github.com/USERNAME/fsdl-text-recognizer-project.git
git push mine master
```

This should prompt you to enter your Github username and password.
If your password is not accepted, it may be because you have two-factor authentication enabled.
Follow directions here to generate a token you can use instead of your password on the command line: https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/
Make sure you store the token (e.g. in your password manager)!

## Setting up Python environment

Now you should be in the repo.

Due to an annoying lack of a feature in pipenv, open Pipfile, and change `tensorflow` to `tensorflow-gpu`.

Run `pipenv install --dev` to install all required packages into a virtual environment.

Make sure to precede all commands with `pipenv run` from now on, to make sure that you are using the correct environment.
(You could run `pipenv shell` to activate the environment in your terminal session, instead.)

## Training the network

You will have to add a tiny bit of code to `text_recognizer/models/mlp.py` before being able to train.
When you finish, you can train a canonical model and save the weights:

```sh
pipenv run training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp"}'
```

You can also run the above command as a shortcut: `tasks/train_character_predictor.py`

Just for fun, you can try a larger MLP, with a larger batch size

```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 8}} "train_args": {"batch_size": 256}'
```

## Testing

Your network is trained, but you need to write a bit more code to get the `CharacterModel` to predict using it.
Open up `text_recognizer/models/character_model.py` and write some code there to make it work.
You can test that it works by running

```sh
pipenv run pytest -s text_recognizer/tests/test_character_predictor.py
```

Or, use the shorthand `tasks/run_prediction_tests.sh`

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
