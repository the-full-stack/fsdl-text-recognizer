## Now

- [x] output solutions in subset_repo
- [-] subset EMNIST so that it's balanced but still has 65 characters
    - Tried, but I don't think it's any better (lower final accuracy)

- [x] create repo that people can clone, add their code, and submit their fork to gradescope
    - [x] make sure can update contents of repo easily with `admin/tasks/subset_repo_into_labs.sh` script
    - make sure that can submit repo to autograder

- [ ] write clear (but still minimal) instructions for what they are supposed to do in lab 1, 2, and 3

- [ ] write files to discuss for every lab, and new lines that appear, for myself as notes

- [] make sure lab3 works

- [o] @arjun @ibrahim the lab3 evaluator runs out of memory
    - Ibrahim is going to give me an option to go up to 3GB

- [x] push base autograder image that has emnist_lines and iam_lines in processed

- [ ] make sure lab4 works

- [ ] make sure lab6 works (circleci)

- [ ] make sure lab7 works

- [ ] make sure lab5 works (probably need to solve the problem saving files)

## Next

- [ ] make a little slide for my approach to sliding window cnn

- [ ] save experiment json along with weights, and just call it canonical_character_predictor_weights.py and canonical_character_predictor_config.py
    - easiest way to implement would probably be to pass in experiment_config from run_experiment to Model#save_weights
- [ ] have networks take more arguments, like lstm_dim and stuff

- [ ] @saurabh support variable-width image input in the LSTM model

- [ ] @arjun don't copy the trained weights over to lab1, but do copy it over to solutions

- [ ] figure out wandb situation: why does it need to be checked into git? (if it's not, message to `wandb init` pops up every time)

- [ ] @saurabh clean up notebooks: don't need as many, just need to show data and some training

- [ ] improve accuracy everywhere we can

- [ ] run a big sample experiment on the farm to generate data that i can show off on W&B

- [ ] find solution to tensorflow vs tensorflow-gpu in Pipfile
    - could set it to tensorflow-gpu in subset_repo, but then have to re-generate Pipfile

- [ ] try using only lowercase characters: that will make emnist a better training experience, and probably boost emnistlines accuracies, too
    - [ ] or, can try class_weight in training

## Arjun


Thursday:
- [ ] go through everything as an attendee and make sure it all works

Friday night:
- [ ] get `sls deploy` working from api/ on jupyterhub (see email thread with Chris)

Saturday:
- [ ] make IAM users for the deployment stuff tomorrow
- [ ] decide where to put AWS credit

- [ ] set up basic monitoring for the lambda functions
- [ ] set up data distribution shift monitoring (just by logging the average value of input and creating a LogMetric for it)
- [ ] test it out with the `admin/endpoint_tester` script that John wrote
- [ ] test out cursive.ai and see if the image processing actually works
    - [ ] then we need to support variable-length inputs

## Later

- [ ] make pre-processing logic in predict() and fit()/evaluate() should be shared
- [ ] store dataset over multiple files
