# Lab 6/7: Deployment

## Serving predictions from a web server

First, we will get a Flask web server up and running and serving predictions.

```
pipenv run python api/app.py
```

Open up another terminal tab (click on the '+' button under 'File' to open the
launcher). In this terminal, we'll send some a test image to the web server
we're running in the first terminal. Make sure to `cd` into the `lab6` directory
in this new terminal.

```
export API_URL=http://0.0.0.0:8000
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -w0 -i text_recognizer/tests/support/emnist_lines/or\ if\ used\ the\ results.png)'" }'
```

If you want to look at the image you just sent, you can navigate to
`lab6/text_recognizer/tests/support/emnist_lines` in the file browser on the
left, and open the image.

We can also send a request specifying a URL to an image:
```
curl "${API_URL}/v1/predict?image_url=http://s3-us-west-2.amazonaws.com/fsdl-public-assets/emnist_lines/or%2Bif%2Bused%2Bthe%2Bresults.png"
```

You can shut down your flask server now.
<!-- If instantiated with `IamLinesDataset`

curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -i text_recognizer/tests/support/iam_lines/He\ rose\ from\ his\ breakfast-nook\ bench.png)'" }' -->

## Running web server in Docker

Now, we'll build a docker image with our application. Docker can be a
convenient way to package up your application with all of its dependencies so
it can be easily deployed. The Dockerfile in `api/Dockerfile` defines how we're
building the docker image.

Still in the `lab6` directory, run:

```sh
tasks/build_api_docker.sh
```

Then you can run the server as

```sh
docker run -p 8000:8000 --name api -it --rm text-recognizer-api
```

If needed, you can connect to a running server by doing

TODO

```sh
docker exec -it api bash
```

## Lambda deployment

```sh
# First, install dependencies specified in package.json
npm install

# Edit serverless.yml to call the service with your own name
# e.g. text-recognizer-sergeyk

# Then, run this and you should see a message asking you to set up AWS credentials
sls info

# Install your credentials by going to https://379872101858.signin.aws.amazon.com/console and logging in with the email you used to register for this bootcamp and the password that we set for you
# Go to IAM, Users, click on yourself, and Create Access Key. Put the key/secret in the command below
sls config credentials --provider aws --key AKIAIOSFODNN7EXAMPLE --secret wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# Now you should be ready to deploy
pipenv run sls deploy -v
```

# Run flask app locally
PYTHONPATH=.. pipenv run sls wsgi serve

# Test with curl
export API_URL=http://0.0.0.0:5000
export API_URL="https://1klwfmaohf.execute-api.us-west-2.amazonaws.com/dev/"
curl "${API_URL}/v1/predict?image_url=https://s3-us-west-2.amazonaws.com/fsdl-public-assets/0.png"
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -i ../text_recognizer/tests/support/emnist/0.png)'" }'
```
