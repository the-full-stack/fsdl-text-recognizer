# Lambda Deployment

```sh
# Run flask app locally
PYTHONPATH=.. pipenv run sls wsgi serve

# Test with curl
export API_URL=http://0.0.0.0:5000
export API_URL="https://1klwfmaohf.execute-api.us-west-2.amazonaws.com/dev/"
curl "${API_URL}/v1/predict?image_url=https://s3-us-west-2.amazonaws.com/fsdl-public-assets/0.png"
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -i ../text_recognizer/tests/support/emnist/0.png)'" }'
```

# Lambda Deployment (OLD)

```bash
# Build the package without deploying it
sls package -v

# See what is in the deployment package
unzip -l .serverless/text-recognizer.zip

# Try running it in Docker
mkdir _temp
unzip .serverless/text-recognizer.zip -d _temp
cd _temp
docker run -v $PWD:/var/task -it lambci/lambda:build-python3.6 bash
python handler.py # in docker

# Deploy the whole thing, including CloudFormation stacks
sls deploy -v

# Deploy only the function zip file (faster)
sls deploy function -f predict

# Test the deployed function
sls invoke -f predict -l --data '{"image_url": "https://s3-us-west-2.amazonaws.com/fsdl-public-assets/0.png"}'

# Or test it via curl
https://s3-us-west-2.amazonaws.com/fsdl-public-assets/0.png
