## Now


- [ ] go through labs 1-3 as an attendee and make sure it all works

- [ ] make a little slide for my approach to sliding window cnn (there's one in the vision applications lecture)

## Next

@arjun
- [ ] make sure lab5 works (probably need to solve the problem of how to save files)
- [ ] save experiment json along with weights, and just call it canonical_character_predictor_weights.py and canonical_character_predictor_config.py
    - easiest way to implement would probably be to pass in experiment_config from run_experiment to Model#save_weights


- [ ] make sure lab4 works on jupyterlab (for a fresh user)

- [ ] have networks take more arguments, like lstm_dim and stuff

- [ ] make sure lab6 works (circleci)
    - [x] get circleci humming again
    - [ ] write instructions for lab6

- [ ] make sure lab7 works
    - [ ] get flask and lambda to be working again on basic CharacterPredictor
    - [ ] use LinePredictor and update test scripts to CURL sample line files

## Arjun

- [ ] go through everything as an attendee and make sure it all works

- [ ] make IAM users for the deployment stuff Sunday
    - would be nice if they could log in to console to see the metrics of their lambda function

- [ ] apply AWS credit to team@fullstackdeeplearning.com account

- [ ] deploy your own lambda (either from lab6) or just from the main repo

- [ ] set up data distribution shift monitoring (just by logging the average value of input and creating a LogMetric for it)

- [ ] test the logging out with the `admin/endpoint_tester` script

Nice-to-have:
- [ ] test out cursive.ai image splitting into lines and see if it actually works from a phone picture

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
- [x] @saurabh be able to accept variable-length image inputs (at least for the lstm_ctc model)
- [x] be ready with a backup plan of running stuff on the farm
- [x] write clear (but still minimal) instructions for what they are supposed to do in lab 1, 2, and 3
- [x] don't copy the trained weights over to lab1, but do copy it over to solutions
