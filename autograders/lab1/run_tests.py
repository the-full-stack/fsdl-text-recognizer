import unittest
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover('tests/test_emnist_mlp.py')
    JSONTestRunner(visibility='visible').run(suite)
