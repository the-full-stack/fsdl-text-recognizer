import base64
import json
import os
import glob
import tempfile
from time import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
import requests

import text_region_extractor as tre
import util


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image = util.read_b64_image(data['image'], grayscale=True)

    processed_image = _process_image(image)[0]

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
    """Expects images with text on paper background."""
    thresholded_image = tre.threshold_text_image(image, blur_kernel_size=21)
    white_text_image = (255 - thresholded_image).astype('uint8')

    text_span_along_w = tre.get_text_span(white_text_image, axis=0)
    text_span_along_h = tre.get_text_span(white_text_image, axis=1)
    image_to_be_rotated = text_span_along_w < text_span_along_h

    if image_to_be_rotated:
        print('image is rotated')
        white_text_image = np.rot90(white_text_image)

    regions, _ = tre.extract_text_regions(
        white_text_image,
        merge_fully_overlapping_regions=True,
        params=None,
        merge_bboxes_in_line=True
    )

    regions = [util.resize(region, 28 / region.shape[0]) for region in regions]
    return regions


def test_process_image():
    file_dirname = os.path.dirname(os.path.abspath(__file__))
    data_dir = f'{file_dirname}/camera_images'
    image_filenames = glob.glob(f'{data_dir}/*.jpg')

    for image_filename in image_filenames:
        image = util.read_image(image_filename, grayscale=True)
        regions = _process_image(image)
        print(f'For {image_filename}: {len(regions)} text regions found')
        for i, region in enumerate(regions):
            region_filename = os.path.splitext(image_filename)[0] + f'_lineRegion-{i}.jpg'
            util.write_image(region, region_filename)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8000, debug=True)
    test_process_image()
