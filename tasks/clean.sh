#!/bin/sh
find . -name "__pycache__" | xargs rm
rm -r api/.serverless
rm api/.requirements.zip
rm -r data/
