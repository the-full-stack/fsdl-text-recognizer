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

    fd, path = tempfile.mkstemp()
    with open(path, 'wb') as f:
        f.write(base64.b64decode(data['image'].split(',')[1]))
    os.close(fd)

    # from IPython import embed; embed()

    # image = util.read_b64_image(data['image'], grayscale=True)

    pred, conf = predictor.predict(path)

    return jsonify({'pred': str(pred), 'conf': float(conf)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
