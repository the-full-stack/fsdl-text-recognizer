#!/usr/bin/env python
"""
Training script, to be run from the command line.
"""

import argparse
import os

from gpu_manager import GPUManager


def train(args) -> None:
    import wandb
    wandb.init(config=args)

    import datasets
    import models
    import util

    emnist = datasets.EMNIST()
    model = models.create_fc_model(emnist.num_classes, args.fc_size)
    util.train_model(
        model=model,
        x_train=emnist.x_train,
        y_train=emnist.y_train,
        loss='categorical_crossentropy',
        epochs=3,
        batch_size=256,
        gpu_ind=args.gpu
    )
    score = model.evaluate(emnist.x_test, emnist.y_test, verbose=1)
    wandb.log({'test_loss': score[0], 'test_accuracy': score[1]})
    print('Test loss/accuracy:', score[0], score[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="Provide index of GPU to use. Providing -1 will block until there is a free one.")
    parser.add_argument("--fc_size", type=int, default=128)
    args = parser.parse_args()

    if args.gpu < 0:
        manager = GPUManager()
        args.gpu = manager.get_free_gpu()  # Blocks until one is available
        print(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'

    train(args)


if __name__ == '__main__':
    main()
