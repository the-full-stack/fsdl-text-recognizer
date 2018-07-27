#!/usr/bin/env python

import base64
import glob
import os
import time

import grequests

NUM_CALLS = 500 # per each HTTP method
TIMEOUT = 2.0
LOCAL_IMAGE_GLOB = '../text_recognizer/tests/support/emnist/*.png'
ENDPOINTS_FILE = './endpoints.txt'
IMAGE_URLS_FILE = './remote_images.txt'

def url_for_get(api_url, img_url):
    """Returns a url suitable for testing GET."""
    return "%s?image_url=%s" % (api_url.strip('/'), img_url)

def data_for_post(api_url, img_path):
    """Returns data param for testing POST."""
    with open(img_path, 'rb') as f:
        text = base64.b64encode(f.read()).decode('ascii')
    return {'image': "data:image/png;base64,'%s'" % text}

def build_get_calls(api_url, img_urls):
    """Returns frozen GET calls."""
    return [grequests.get(url_for_get(api_url, img_url), timeout=TIMEOUT)
             for img_url in img_urls]

def build_post_calls(api_url, local_images):
    """Returns frozen POST calls."""
    return [grequests.post(api_url, data=data_for_post(api_url, img_path), timeout=TIMEOUT)
             for img_path in local_images]

def main():
    """Reads the files and runs everything."""
    with open(ENDPOINTS_FILE) as endpoints_file:
        endpoints = [x.strip() for x in endpoints_file.readlines()]
    with open(IMAGE_URLS_FILE) as image_urls_file:
        remote_image_urls = [x.strip() for x in image_urls_file.readlines()]
    local_images = glob.glob(LOCAL_IMAGE_GLOB)

    # build set of roughly 200 calls
    stuff = []
    for url in endpoints:
        stuff.extend(build_get_calls(url, remote_image_urls))
        stuff.extend(build_post_calls(url, local_images))
    stuff *= int(200 / len(stuff))

    good = 0
    total = 0
    while True:
        responses = grequests.map(stuff)
        total += len(stuff)
        good += (len(stuff) - responses.count(None))
        b = "%s of %s completed." % (good, total)
        print (b, end="\r")

if __name__ == '__main__':
    main()
