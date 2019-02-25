#!/bin/bash
set -euo pipefail

echo "pipenv check"
pipenv check

echo "pylint"
pipenv run pylint api text_recognizer training

echo "pycodestyle"
pipenv run pycodestyle api text_recognizer training

echo "mypy"
pipenv run mypy api text_recognizer training

echo "bandit"
pipenv run bandit -ll -r {api,text_recognizer,training}

echo "shellcheck"
shellcheck ./**/*.sh
