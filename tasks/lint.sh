#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "pipenv check"
pipenv check  # Not reporting failure here, because sometimes this fails due to API request limit

echo "pylint"
pylint --ignore=.serverless api text_recognizer training || FAILURE=true

echo "pycodestyle"
pycodestyle --exclude=node_modules,.serverless,.ipynb_checkpoints api text_recognizer training || FAILURE=true

# echo "pydocstyle"
# pydocstyle pandagrader projects lambda_deployment || FAILURE=true

echo "mypy"
mypy api text_recognizer training || FAILURE=true

echo "bandit"
bandit -ll -r {api,text_recognizer,training} -x node_modules,.serverless || FAILURE=true

echo "shellcheck"
shellcheck tasks/*.sh || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0
