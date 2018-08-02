## Now

- [x] output solutions in subset_repo

- [ ] subset EMNIST so that it's balanced but still has 65 characters

- [ ] create repo that people can clone, add their code, and submit their fork to gradescope
    - [x] make sure can update contents of repo easily with `admin/tasks/subset_repo_into_labs.sh` script
    - make sure that can submit repo to autograder

- [ ] write clear (but still minimal) instructions for what they are supposed to do in lab 1, 2, and 3

- [ ] write files to discuss for every lab, and new lines that appear, for myself as notes

## Next

- [ ] save experiment json along with weights, and just call it canonical_character_predictor_weights.py and canonical_character_predictor_config.py
    - easiest way to implement would probably be to pass in experiment_config from run_experiment to Model#save_weights
- [ ] Support variable-width image input in the LSTM model
- [ ] have networks take more arguments, like lstm_dim and stuff
- [ ] clean up notebooks: don't need as many, just need to show data and some training
- [ ] try using only lowercase characters

## Arjun

Thursday:
- [ ] go through everything as an attendee and make sure it all works

Friday night:
- [ ] get `sls deploy` working from api/ on jupyterhub (see email thread with Chris)

Saturday:
- [ ] set up basic monitoring for the lambda functions
- [ ] set up data distribution shift monitoring (just by logging the average value of input and creating a LogMetric for it)
- [ ] test it out with the `admin/endpoint_tester` script that John wrote
- [ ] test out cursive.ai and see if the image processing actually works
    - [ ] then we need to support variable-length inputs

## Later

- [ ] make pre-processing logic in predict() and fit()/evaluate() should be shared
- [ ] store dataset over multiple files
