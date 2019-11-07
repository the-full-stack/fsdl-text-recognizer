FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install pipenv

RUN pipenv sync -d
