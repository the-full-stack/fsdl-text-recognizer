FROM ubuntu:18.04

# Install pipenv
RUN apt-get update && apt-get install python3-pip -y && pip3 install pipenv
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Copy build dir to /app
RUN set -ex && mkdir /app
WORKDIR /app
COPY . .

# Install Python dependencies
RUN set -ex && pipenv sync --deploy --system

# Run the web server
ENV PYTHONPATH /app
CMD python3 /app/text_recognizer/web/api.py
