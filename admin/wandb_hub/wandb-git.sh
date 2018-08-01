#!/bin/bash
BRANCH=${1:-master}
pip install --upgrade git+git://github.com/wandb/client.git@$BRANCH#egg=wandb