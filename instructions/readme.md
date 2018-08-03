# Full Stack Deep Learning Labs

Welcome!

## W&B Jupyter Hub instructions

Go to https://app.wandb.ai/profile and enter the code that we will share with you at the session into Access Code field.
This will drop you into a JupyterLab instance with a couple of GPUs that you will use in these labs.

**From now on, everything you do will be in that instance.**

## Checking out the repo

Start by cloning the repo and going into it

```
git clone https://github.com/gradescope/fsdl-text-recognizer-project.git
cd fsdl-text-recognizer-project
```

If you already have the repo in your home directory, then simply go into it and pull the latest version.

```sh
cd fsdl-text-recognizer-project
git pull origin master
```

Now click Fork in the top right of this Github repo page: https://github.com/gradescope/fsdl-text-recognizer-project.
Select your personal account to fork to, and note down your USERNAME, which will be right after https://github.com in the URL that you will be taken to.

Add your fork as a remote of the repo and push to it:

```sh
git remote add mine https://github.com//fsdl-text-recognizer-project.git
git push mine master
```

(If you face some kind of issue, you can `git remote rm mine` and then add it again.)

Pushing will prompt you to enter your Github username and password.
If your password is not accepted, it may be because you have two-factor authentication enabled.
Follow directions here to generate a token you can use instead of your password on the command line: https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/
Make sure you store the token (e.g. in your password manager), as you will not be able to see it again.
If you want to avoid entering the token with each git push, you can run the following command to cache your credentials:

```sh
git config --global credential.helper cache
```

Note that if you do cache your credentials, we recommend you delete this token after the bootcamp.

## Setting up the Python environment

Run `pipenv install --dev` to install all required packages into a virtual environment.

Make sure to precede all commands with `pipenv run` from now on, to make sure that you are using the correct environment.
Or, you could run `pipenv shell` to activate the environment in your terminal session, instead.
Remember to do that in every terminal session you start.

## Ready

Now you should be setup for the labs. The instructions for each lab are in readme files in their folders.

You will notice that there are solutions for all the labs right here in the repo, too.
If you get stuck, you are welcome to take a look!
