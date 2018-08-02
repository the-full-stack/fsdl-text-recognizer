#!/bin/sh
# Run this file from the top-level repo.

# Build base image for all autograders
docker build -t sergeykarayev/fsdl-text-recognizer-labs:base -f admin/autograders/base/Dockerfile .
docker push sergeykarayev/fsdl-text-recognizer-labs:base

# Lab 1 (MLP)
docker build -t sergeykarayev/fsdl-text-recognizer-labs:lab1 -f admin/autograders/lab1/Dockerfile .
docker push sergeykarayev/fsdl-text-recognizer-labs:lab1
# docker run -it -v $PWD/text_recognizer:/autograder/source/text_recognizer sergeykarayev/fsdl-text-recognizer-labs:lab1 bash

# # Lab 2 (CNN)
# docker build -t sergeykarayev/fsdl-text-recognizer-labs:lab2 -f admin/autograders/lab2/Dockerfile .
# docker push sergeykarayev/fsdl-text-recognizer-labs:lab2
# # docker run -it -v $PWD/text_recognizer:/autograder/source/text_recognizer sergeykarayev/fsdl-text-recognizer-labs:lab2 bash

# Lab 3 (LSTM)
docker build -t sergeykarayev/fsdl-text-recognizer-labs:lab3 -f admin/autograders/lab3/Dockerfile .
docker push sergeykarayev/fsdl-text-recognizer-labs:lab3
# docker run -it -v $PWD/text_recognizer:/autograder/source/text_recognizer sergeykarayev/fsdl-text-recognizer-labs:lab3 bash

# Lab 4 (W&B) has no autograder

# Lab 5 (Free)
docker build -t sergeykarayev/fsdl-text-recognizer-labs:lab5 -f admin/autograders/lab5/Dockerfile .
docker push sergeykarayev/fsdl-text-recognizer-labs:lab5
# docker run -it -v $PWD/text_recognizer:/autograder/source/text_recognizer sergeykarayev/fsdl-text-recognizer-labs:lab5 bash

# # Lab 6 (CircleCI, Flask)
# docker build -t sergeykarayev/fsdl-text-recognizer-labs:lab6 -f admin/autograders/lab6/Dockerfile .
# docker push sergeykarayev/fsdl-text-recognizer-labs:lab6

# Lab 7 (Lambda) has no autograder
