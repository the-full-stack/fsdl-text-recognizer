# FSDL Text Recognizer Autograders

The base image:
- installs python3-pip, pipenv, and all dependencies
- copies data/processed (for running evaluation)
- copies text_recognizer/tests (for running unit tests)

The lab-specific images start from the base image and simply copy lab-specific `run_tests.py` files.

## Testing locally

```sh
docker run -it -v $PWD/text_recognizer:/autograder/source/text_recognizer sergeykarayev/fsdl-text-recognizer-labs:lab1 bash
```
