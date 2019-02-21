#!/bin/sh
find . -name "__pycache__" | xargs rm -r
rm -r api/.serverless
rm api/.requirements.zip
rm -r data/
