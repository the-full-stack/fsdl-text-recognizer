## Now

- [ ] make sure that every lab runs on JupyterHub

- [ ] write clear (but still minimal) instructions for what they are supposed to do in lab 1, 2, and 3

- [ ] write files to discuss for every lab, and new lines that appear, for myself as notes

- [ ] make sure lab4 works on jupyterlab

- [ ] make sure lab6 works (circleci)
    - [x] get circleci humming again
    - [ ] write instructions for lab6

- [ ] make sure lab7 works
    - [ ] get flask and lambda to be working again on basic CharacterPredictor
    - [ ] use LinePredictor and update test scripts to CURL sample line files

- [ ] make sure lab5 works (probably need to solve the problem saving files)

- [ ] @saurabh be able to accept variable-length image inputs (at least for the lstm_ctc model)
    - done, need to verify that it works

- [ ] @saurabh improve accuracy everywhere we can

## Next

- [ ] @arjun have a plan for dealing with merge conflicts in fsdl-text-recognition-project

- [ ] make a little slide for my approach to sliding window cnn

@arjun
- [ ] save experiment json along with weights, and just call it canonical_character_predictor_weights.py and canonical_character_predictor_config.py
    - easiest way to implement would probably be to pass in experiment_config from run_experiment to Model#save_weights

- [ ] @arjun don't copy the trained weights over to lab1, but do copy it over to solutions

- [ ] have networks take more arguments, like lstm_dim and stuff

- [ ] figure out wandb situation: why does it need to be checked into git? (if it's not, message to `wandb init` pops up every time)

## Arjun


- [ ] go through everything as an attendee and make sure it all works

- [ ] make IAM users for the deployment stuff Sunday
- [ ] decide where to put AWS credit (team@fullstackdeeplearning.com)?

- [ ] set up basic monitoring for the lambda functions
- [ ] set up data distribution shift monitoring (just by logging the average value of input and creating a LogMetric for it)
- [ ] test it out with the `admin/endpoint_tester` script that John wrote
- [ ] test out cursive.ai and see if the image processing actually works
    - [ ] then we need to support variable-length inputs

## Later

- [ ] make pre-processing logic in predict() and fit()/evaluate() should be shared
- [ ] store dataset over multiple files


## Done

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
