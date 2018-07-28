# https://github.com/UnitedIncome/serverless-python-requirements
try:
  import unzip_requirements
except ImportError:
  pass

import base64
import os
import tempfile

import flask
from flask import Flask, request, jsonify
import numpy as np

from text_recognizer.character_predictor import CharacterPredictor
import text_recognizer.util as util

app = Flask(__name__)
predictor = CharacterPredictor()


@app.route('/')
def index():
    return 'Hello, world!'


@app.route('/v1/predict', methods=['GET', 'POST'])
def predict():
    image = _load_image()
    pred, conf = predictor.predict(image)
    return jsonify({'pred': str(pred), 'conf': float(conf)})


def _load_image():
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            return 'no json received'
        return util.read_b64_image(data['image'], grayscale=True)
    elif request.method == 'GET':
        image_url = request.args.get('image_url')
        if image_url is None:
            return 'no image_url defined in query string'
        return util.read_image(image_url, grayscale=True)
    else:
        raise ValueError('Unsupported HTTP method')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
