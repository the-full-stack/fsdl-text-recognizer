## Next

- [ ] 1 go through all labs in Jupyterhub and take screenshots, putting them into the readme
- [ ] 1 make the app.py use the joint model

- [ ] 2 add more information to slides as preview of the important things we'll be doing
  - [ ] ability to run end-to-end from raw data, with caching along the way to speed up future runs
  - [ ] dataset streaming and augmentations (fast.ai, TFRecord)
  - [ ] specifying and recording experiments via config file
  - [ ] ability to run experiments and automatically pick best model
  - [ ] ability to create a deployment package in CI
- [ ] 2 introduce code that picks best run from weights and biases (2 hours)
- [ ] 2 add multi-gpu data parallelism option in run_experiment.py
- [ ] 2 look into switching from flask to that async one in fast.ai course
- [ ] 2 kick off another IAM training with ImageDataGenerator
- [ ] 2 add tests for training (but don't run them in circleci)
- [ ] 2 add to lab 4: output sample predictions every epoch so that they can be reviewed in weights and biases
- [ ] 2 save experiment json along with weights, and just call it canonical_character_predictor_weights.h5 and canonical_character_predictor_config.json
    - easiest way to implement would probably be to pass in experiment_config from run_experiment to Model#save_weights

- [ ] 3 add a notebook that uses our trained line detector on the fsdl handwriting data
- [ ] 3 share pre-processing logic in predict() and fit()/evaluate()
- [ ] 3 compute validation accuracy in ctc training (run decoding)

- [ ] 4 make a flag for overfitting on one batch

## Done

- [x] 20191029 look into writing lab readme's as slides using Marp, but decided against it for now, because wasn't able to find a solution that looked good in both github readme format (and typora) and marp
- [x] 20191030 1 update Pipfile
  - tensorflow 1.15 seems to depend on functools32 which can't be installed for python3
  - tensorflow 1.14 has the dual -gpu and not-gpu nature, which is a little annoying, but fine
  - tensorflow 2.0 also has dual gpu
  - python3.7 has trouble installing a dependency of wandb (forgot the name)
  - settled on python3.6 and tensorflow 1.14
