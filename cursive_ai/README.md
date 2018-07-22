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
curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -i 0.png)'" }'
```

## File Upload

Interesting:
- https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594

## HTML

- Can spruce the file upload button up with https://codepen.io/stebaker/pen/tImBc
