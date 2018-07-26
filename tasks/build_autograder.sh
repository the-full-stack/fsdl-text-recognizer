#!/bin/sh

# Build base image
docker build -t sergeykarayev/fsdl-text-recognizer-labs:base -f autograders/base/Dockerfile .
docker push sergeykarayev/fsdl-text-recognizer-labs:base
# Test:
# docker run -it -v $PWD/text_recognizer:/autograder/source/text_recognizer sergeykarayev/fsdl-text-recognizer-labs:base bash

# Build Lab 1 autograder
docker build -t sergeykarayev/fsdl-text-recognizer-labs:lab1 -f autograders/lab1/Dockerfile .
docker push sergeykarayev/fsdl-text-recognizer-labs:lab1
