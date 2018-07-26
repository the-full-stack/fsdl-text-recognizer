class LineCnnSlidingWindow(LineModel):
    def __init__(self, window_fraction: float=0.5, window_stride: float=0.5):
        super().__init__()
        self.window_fraction = window_fraction
        self.window_stride = window_stride

    @cachedproperty
    def model(self):
        return create_sliding_window_image_model(self.input_shape, self.max_length, self.num_classes, self.window_fraction, self.window_stride)


def create_sliding_window_image_model(
        image_shape: Tuple[int, int],
        max_length: int,
        num_classes: int,
        window_width_fraction: float=0.5,
        window_stride_fraction: float=0.5) -> KerasModel:
    image_height, image_width = image_shape
    image_input = Input(shape=image_shape)

    letter_width = image_width // max_length
    window_width = int(letter_width * window_width_fraction)
    window_stride = int(letter_width * window_stride_fraction)
    def slide_window(image, window_width=window_width, window_stride=window_stride):
        # import tensorflow as tf  # Might need this import for Keras to save/load properly
        # batch_size, image_height, image_width, num_channels = image.shape
        kernel = [1, 1, window_width, 1]
        strides = [1, 1, window_stride, 1]
        patches = tf.extract_image_patches(image, kernel, strides, [1, 1, 1, 1], 'SAME')
        patches = tf.transpose(patches, (0, 2, 1, 3))
        patches = tf.expand_dims(patches, -1)
        return patches

    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    image_patches = Lambda(slide_window)(image_reshaped)  # (num_windows, num_windows, window_width, 1)

    convnet = lenet(image_height, window_width)  # Note that this doesn't include the top softmax layer
    convnet_outputs = TimeDistributed(convnet)(image_patches)  # (num_windows, 128)
    convnet_outputs_extra_dim = Lambda(lambda x: tf.expand_dims(x, -1))(convnet_outputs) # (num_windows, 128, 1)

    width = int(1 / window_stride_fraction)
    conved_convnet_outputs = Conv2D(num_classes, (width, 128), (width, 1), activation='softmax')(convnet_outputs_extra_dim) # (max_length, 1, num_classes)
    conved_convnet_outputs_squeezed = Lambda(lambda x: tf.squeeze(x, 2))(conved_convnet_outputs) # (max_length, num_classes)

    model = KerasModel(inputs=image_input, outputs=conved_convnet_outputs_squeezed)
    model.summary()
    return model
