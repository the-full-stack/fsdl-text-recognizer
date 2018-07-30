#!/bin/sh
pipenv lock --requirements --keep-outdated > api/requirements.txt
cd api
sls plugin install -n serverless-python-requirements
sls plugin install -n serverless-wsgi
sls deploy -v
