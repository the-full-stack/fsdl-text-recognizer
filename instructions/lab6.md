# Lab 6: Line Detection

## Looking at the data

- Look at `notebooks/04-look-at-iam-paragraphs.ipynb`

## Data processing

## Network description

- Look at `text_recognizer/networks/fcn.py`.

The basic idea is a deep convolutional network with resnet-style blocks (input to block is concatenated to block output).
We call it FCN, as in "Fully Convolutional Network," after the seminal paper that first used convnets for segmentation.

Unlike the original FCN, however, we do not maxpool or upsample, but instead rely on dilated convolutions to rapidly increase the effective receptive field.
With `padding='SAME'`, stacking conv layers results in an output that is exactly the same size as the image, which is what we want.
[Here](https://fomoro.com/projects/project/receptive-field-calculator) is a very calculator of the effective receptive field size of a convnet.

The crucial thing to understand is that because we are labeling odd and even lines differently, each predicted pixel must have the context of the entire image to correctly label -- otherwise, there is no way to know whether the pixel is on an odd or even line.

## Data augmentation

Because we only have about a thousand images to learn this task on, data augmentation will be crucial.
Image augmentations such as streching, slight rotations, offsets, contrast and brightness changes, and potentially even mirror-flipping are tedious to code up ourselves, and most frameworks provide a utility for doing it.

We use Keras's `ImageDataGenerator`, and you can see the parameters for it in `text_recognizer/models/line_detector_model.py`.

## Review training results

## Combining the two models

## Things to try

- Try adding more data augmentations, or mess with the parameters of the existing ones
