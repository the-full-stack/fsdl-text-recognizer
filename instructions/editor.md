# Setting up editor

## VSCode

There are two things you want to make sure of when using VSCode: 1) that it uses the right environment, and 2) that it lints your files as you work.

Here is my setup for linting:

```
{
  "editor.rulers": [120],
  "files.exclude": {
    "**/.git": true,
    "**/.DS_Store": true,
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true
  },
  "python.linting.pep8Enabled": true,
  "python.linting.pep8Path": "pycodestyle",
  "python.linting.pylintEnabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.banditEnabled": true,
  "python.linting.banditArgs": ["-ll"],
  "python.linting.enabled": true,
  "[python]": {
    "editor.tabSize": 4
  },
}

```
