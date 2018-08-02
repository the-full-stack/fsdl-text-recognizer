## Now

- [x] hide future labs from current lab
- [x] upload emnist matlab.zip to S3 to that it's faster to download

- [ ] is it easy to support variable-length inputs?

## Arjun

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
