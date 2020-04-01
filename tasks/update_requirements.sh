#!/bin/bash

pip-compile -v requirements.in && pip-compile -v requirements-dev.in
