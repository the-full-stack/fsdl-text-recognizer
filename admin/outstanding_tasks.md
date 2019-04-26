## Next

- [ ] 1 make the app.py use the joint model

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
