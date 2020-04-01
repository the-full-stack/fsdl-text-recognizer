FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt
