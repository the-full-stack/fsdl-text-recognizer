## Next

- If I could get marp to work with <img> tags, it would be great. For now, just bust through these tasks without worrying about it
  - FOR NOW just make one big keynote presentation for all labs

- [ ] 1 before changing anything else, go through all labs in Jupyterhub and take screenshots, putting them into the readme
- [ ] 1 make the app.py use the joint model

Josh could do:
- [ ] 1 add more information to intro slides as preview of the important things we'll be doing
  - [ ] ability to run end-to-end from raw data, with caching along the way to speed up future runs
  - [ ] dataset streaming and augmentations (fast.ai, TFRecord)
  - [ ] specifying and recording experiments via config file
  - [ ] ability to run experiments and automatically pick best model
  - [ ] ability to create a deployment package in CI

- [ ] 2 introduce code that picks best run from weights and biases (2 hours)
- [ ] 2 introduce config.json in running experiments (4 hours)
- [ ] 2 explain pipenv in readme (1 hour)

- [ ] 2 add data parallelism option

- [ ] 2 add a notebook that uses our trained line detector on the fsdl handwriting data
- [ ] 2 kick off another IAM training with ImageDataGenerator
- [ ] 2 add tests for training (but don't run them in circleci)
- [ ] 2 add to lab 5: output sample predictions every epoch so that they can be reviewed in weights and biases

- [ ] 2 save experiment json along with weights, and just call it canonical_character_predictor_weights.h5 and canonical_character_predictor_config.json
    - easiest way to implement would probably be to pass in experiment_config from run_experiment to Model#save_weights

- [ ] 3 share pre-processing logic in predict() and fit()/evaluate()
- [ ] 3 compute validation accuracy in ctc training (run decoding)

- [ ] 4 make a flag for overfitting on one batch
- [ ] 4 add metadata.toml for Brown corpus
- [ ] 4 develop code to create IAM lines from IAM source data
    - [ ] load in train/val/test ids in IamDataset

## Done

- [x] 20191029 look into writing lab readme's as slides using Marp, but decided against it for now, because wasn't able to find a solution that looked good in both github readme format (and typora) and marp
- [x] 20191030 1 update Pipfile
  - tensorflow 1.15 seems to depend on functools32 which can't be installed for python3
  - tensorflow 1.14 has the dual -gpu and not-gpu nature, which is a little annoying, but fine
  - tensorflow 2.0 also has dual gpu
  - python3.7 has trouble installing a dependency of wandb (forgot the name)
  - settled on python3.6 and tensorflow 1.14
