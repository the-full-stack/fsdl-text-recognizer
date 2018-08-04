# Start from the latest Long Term Support (LTS) Ubuntu version
FROM ubuntu:18.04

# Install pipenv
RUN apt-get update && apt-get install python3-pip -y && pip3 install pipenv

# Copy build dir to /repo
RUN set -ex && mkdir /repo
WORKDIR /repo
COPY text_recognizer/ .
COPY api/ .
COPY Pipfile .
COPY Pipfile.lock .

# Install Python dependencies
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN set -ex && pipenv install --keep-outdated --deploy --system

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 /repo/api/app.py
