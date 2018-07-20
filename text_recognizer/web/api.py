import base64
import os
import tempfile

import flask
from flask import Flask, request, jsonify
import numpy as np

from text_recognizer.emnist_mlp_predictor import EmnistMlpPredictor
import text_recognizer.util as util

app = Flask(__name__)
predictor = EmnistMlpPredictor()


@app.route('/')
def index():
    return 'Hello, world!'


@app.route('/v1/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if data is None:
        return 'no json received'
    image = util.read_b64_image(data['image'], grayscale=True)
    pred, conf = predictor.predict(image)
    return jsonify({'pred': str(pred), 'conf': float(conf)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
