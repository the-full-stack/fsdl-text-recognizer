#!/bin/bash

find . -name "__pycache__" -exec rm -r {} \;
rm -r api/.serverless
rm api/.requirements.zip
rm -r data/
