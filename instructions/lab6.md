# Lab 6/7: Deployment

## Serving predictions from a web server

First, we will get a Flask web server up and running and serving predictions.

```
pipenv run python api/app.py
```

Now we can send some test images to it

```
export API_URL=http://0.0.0.0:8000
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -i text_recognizer/tests/support/emnist_lines/or\ if\ used\ the\ results.png)'" }'
```

<!-- If instantiated with `IamLinesDataset`

curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -i text_recognizer/tests/support/iam_lines/He\ rose\ from\ his\ breakfast-nook\ bench.png)'" }' -->

## Running web server in Docker

Execute this from top-level repo:

```sh
docker build -t text-recognizer-api -f api/Dockerfile .
```

Then you can run the server as

```sh
docker run -p 8000:8000 --name api -it text-recognizer-api
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
