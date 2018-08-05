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

## Adding tests for web server

The web server code should have a unit test just like the rest of our code.

Let's check it out: the tests are in `api/tests/test_app.py`, and you can run them with `tasks/test_api.sh`

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
docker run -p 8000:8000 --name api -it --rm text_recognizer_api
```

You can run the same curl commands as you did when you ran the flask server earlier, and see that you're getting the same results.

If needed, you can connect to your running docker container by running:

```sh
docker exec -it api bash
```

You can shut down your docker container now.

We could deploy this container to, for example, AWS Elastic Container Service.
Feel free to do that as an exercise after the bootcamp!

In this lab, we will deploy the app as a package to AWS Lambda.

## Lambda deployment

To deploy to AWS Lambda, we ar egong to use the `serverless` framework.

First, `cd` into the `lab6/api` directory and install the dependencies for serverless:

```sh
npm install
```

Next, we'll need to configure serverless. Edit `serverless.yml` and change the service name on the first line (you can use your Github username for USERNAME):

```
service: text-recognizer-USERNAME
```

Next, run `sls info`. You'll see a message asking you to set up your AWS credentials. We sent an email to you with your AWS credentials (let us know if you can't find it).
Note that emailing credentials is generally a bad idea. You usually want to handle credentials in a more secure fashion.
We're only doing it in this case because your credentials give you very limited access and are for a temporary AWS account.

You can also go to https://379872101858.signin.aws.amazon.com/console and log in with the email you used to register (and the password we emailed you), and create your own credentials if you prefer.

Edit the command below and substitute your credentials for the placeholders:

```
sls config credentials --provider aws --key AKIAIOSFODNN7EXAMPLE --secret wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

Now you've got everything configured, and are ready to deploy. Serverless will package up your flask API before deploying it.
It will install all of the python packages in a docker container that matches the environment lambda uses, to make sure the compiled code is compatible.
This will take 3-5 minutes. This command will package up and deploy your flask API:

```
pipenv run sls deploy -v
```

Near the end of the output of the deploy command, you'll see links to your API endpoint. Copy the top one (the one that doesn't end in `{proxy+}`).

As before, we can test out our API by running a few curl commands (from the `lab6` directory). We need to change the `API_URL` first though to point it at Lambda:

```
export API_URL="https://REPLACE_THIS.execute-api.us-west-2.amazonaws.com/dev/"
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -w0 -i text_recognizer/tests/support/emnist_lines/or\ if\ used\ the\ results.png)'" }'
curl "${API_URL}/v1/predict?image_url=http://s3-us-west-2.amazonaws.com/fsdl-public-assets/emnist_lines/or%2Bif%2Bused%2Bthe%2Bresults.png"
```

You'll want to run the curl commands a couple of times -- the first execution will take much longer than the second, because the function has to "warm up."
After the first request, it will stay warm for 10-60 minutes.

In addition to deploying to AWS, serverless lets you test out everything locally as well.
We'll make sure everything works locally first. We're going to use serverless to run the flask API locally:
We have to have already run `sls deploy`, because this relies on the package being created already.

```sh
PYTHONPATH=.. pipenv run sls wsgi serve
```

Again, we can test out our API by running a few curl commands after changing the `API_URL` (from the `lab6` directory):

```
export API_URL=http://0.0.0.0:5000
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -w0 -i text_recognizer/tests/support/emnist_lines/or\ if\ used\ the\ results.png)'" }'
curl "${API_URL}/v1/predict?image_url=http://s3-us-west-2.amazonaws.com/fsdl-public-assets/emnist_lines/or%2Bif%2Bused%2Bthe%2Bresults.png"
```
