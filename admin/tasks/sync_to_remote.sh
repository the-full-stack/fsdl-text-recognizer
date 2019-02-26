#!/bin/bash

rsync -aP --exclude=".*" --exclude=data --exclude=wandb --exclude=node_modules . thefarm:work/fsdl-text-recognizer/
