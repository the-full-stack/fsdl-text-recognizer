- I'll have to figure out https://serverless.com/blog/cors-api-gateway-survival-guide/

## Setting up

```sh
sls create -t aws-python3 -n cursive-ai
sls plugin install -n serverless-wsgi
sls plugin install -n serverless-python-requirements
# edit serverless.yml and app.py
pipenv install
pipenv shell
sls wsgi serve
# see stuff work on localhost
```

## Testing with CURL

```sh
curl -X POST https://r0bhgvb5y7.execute-api.us-west-2.amazonaws.com/dev/predict -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -i ../text_recognizer/tests/support/emnist/0.png)'", "predict_api_url": "https://1klwfmaohf.execute-api.us-west-2.amazonaws.com/dev/v1/predict" }'
# {"api_response":{"conf":1.0,"pred":"0"},"time_taken":0.08494734764099121}
```

## HTML

- Can spruce the file upload button up with https://codepen.io/stebaker/pen/tImBc
