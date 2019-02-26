## Now

- [ ] go through the first 5 labs and make sure it all works
    - [x] make tests not use GPU
    - [ ] figure out if there's something weird about ctc model (seems to die after four epochs on the farm)
- [ ] add metadata.toml and download data in a separate script, not from dataset python file directly
- [ ] strip &quot character from IAM dataset

## Next

- [ ] add tests for training (but don't run them in circleci)
- [x] add "subsample" mode to dataset
- [ ] add to lab 5: output sample predictions every epoch so that they can be reviewed in weights and biases
- [ ] get the evaluation tests to pass
- [ ] train a baller IAMLines model and use that in app.py
- [ ] explain Pipenv in lab1.md
- [ ] consider using datasets to the training class (move the mappings to models/ then)

- [ ] document my approaches to line detection and save the notebooks, so that saurabh can take over tuesday morning
- [ ] port the IAM forms data to datasets/ and data/raw

- [ ] get a basic version of a data annotation interface working (flask app with json db and react frontend)
- [ ] get a great line detection model going in notebook
- [ ] port line detection model to proper files, train it, and save model
- [ ] connect the line detection and line recognition model
- [ ] add test for the joint model
- [ ] make the app.py use the joint model
- [ ] compute validation accuracy in ctc training (run decoding)

## Ideas

- [ ] save experiment json along with weights, and just call it canonical_character_predictor_weights.py and canonical_character_predictor_config.py
    - easiest way to implement would probably be to pass in experiment_config from run_experiment to Model#save_weights
- [ ] share pre-processing logic in predict() and fit()/evaluate()

## Done

### 2019 Feb 25

- [x] set up linting
    - [x] make the lint.sh script complete even when some commands fail
- [x] get rid of sliding-window cnn
- [x] get rid of non-ctc lstm
- [x] get to ~100% linted
- [x] use git-lfs for models
- [x] add git-lfs to circleci config
- [x] add linting to lab circleci

### 2018 July/August

- [x] data/ autograder image docker thing (the lab code now expects it one level up from where it was)
- [x] output solutions in subset_repo
- [-] subset EMNIST so that it's balanced but still has 65 characters
    - Tried, but I don't think it's any better (lower final accuracy)
- [x] create repo that people can clone, add their code, and submit their fork to gradescope
    - [x] make sure can update contents of repo easily with `admin/tasks/subset_repo_into_labs.sh` script
    - make sure that can submit repo to autograder
- [x] make sure lab3 works
- [o] @arjun @ibrahim the lab3 evaluator runs out of memory
    - Aug 2 14:30 Ibrahim is going to give me an option to go up to 3GB
    - even with 3GB it crashed, but now due to a different error: std:bad_alloc (which is from tensorflow)
        - Ibrahim can make the instances bigger
        - Sergey a small batch_size on the evaluate code
- [x] push base autograder image that has emnist_lines and iam_lines in processed
- [x] @saurabh be able to accept variable-length image inputs (at least for the lstm_ctc model)
- [x] be ready with a backup plan of running stuff on the farm
- [x] write clear (but still minimal) instructions for what they are supposed to do in lab 1, 2, and 3
- [x] don't copy the trained weights over to lab1, but do copy it over to solutions
- [x] go through labs 1-3 as an attendee in jupyterlab and make sure it all works
- [x] push clean commit history
- [x] make sure lab5 works (probably need to solve the problem of how to save files)
- [x] make sure lab4 works on jupyterlab (for a fresh user)
- [x] make sure lab6 works (circleci)
    - [x] get circleci humming again
    - [x] write instructions for lab6
- [x] make sure lab7 works
    - [x] get flask and lambda to be working again on basic CharacterPredictor
    - [x] use LinePredictor and update test scripts to CURL sample line files
- [x] @arjun go through everything as an attendee and make sure it all works
- [x] @arjun make IAM users for the deployment stuff Sunday
- [x] apply AWS credit to team@fullstackdeeplearning.com account
- [x] deploy your own lambda (either from lab6) or just from the main repo
- [x] set up data distribution shift monitoring (just by logging the average value of input and creating a LogMetric for it)
- [x] test the logging out with the `admin/endpoint_tester` script
- [x] make autograder for lab 5: simply use the existing tests for line predictor, and also add a new evaluate function on iam
