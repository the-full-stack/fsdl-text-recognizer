# https://github.com/UnitedIncome/serverless-python-requirements
try:
  import unzip_requirements
except ImportError:
  pass

import os
import sys
import json

from text_recognizer.emnist_mlp_predictor import EmnistMlpPredictor


def predict(event, context):
    """
    This is the function called by AWS Lambda, passing the standard parameters "event" and "context"
    When deployed, you can try it out pointing your browser to
    {LambdaURL}/{stage}/predict?x=2.7
    where {LambdaURL} is Lambda URL as returned by serveless installation and {stage} is set in the
    serverless.yml file.
    """
    try:
        image_url = event['queryStringParameters']['image_url']
        predictor = EmnistMlpPredictor()
        pred, conf = predictor.predict(image_url)
        return lambda_gateway_response(200, {'pred': str(pred), 'conf': float(conf)})
    except Exception as ex:
        error_response = {
            'error_message': "Unexpected error",
            'stack_trace': str(ex)
        }
        return lambda_gateway_response(503, error_response)


def lambda_gateway_response(code, body):
    """
    This function wraps around the endpoint responses in a uniform and Lambda-friendly way
    :param code: HTTP response code (200 for OK), must be an int
    :param body: the actual content of the response
    """
    cors_headers = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Credentials': True
    }
    response = {"statusCode": code, "headers": cors_headers, "body": json.dumps(body)}
    print(response)
    return response


if __name__ == '__main__':
    import pathlib
    predict({'image_url': sys.argv[1]}, {})
