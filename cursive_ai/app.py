import base64
import json
import os
import tempfile
from time import time

from flask import Flask, jsonify, render_template, request
import requests

import util


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image = util.read_b64_image(data['image'], grayscale=True)

    processed_image = _process_image(image)

    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        util.write_image(processed_image, fp.name)
        fp.seek(0)
        processed_image_b64 = base64.b64encode(fp.read())

    t = time()
    api_response = requests.post(
        data['predict_api_url'],
        data=json.dumps({'image': f'data:image/png;base64,{processed_image_b64.decode()}'}),
        headers={'Content-Type': 'application/json'}
    )
    return jsonify({'api_response': api_response.json(), 'time_taken': time() - t})


def _process_image(image):
    if image.shape != (28, 28):
        image = cv2.resize(image, (28, 28))
    return image
