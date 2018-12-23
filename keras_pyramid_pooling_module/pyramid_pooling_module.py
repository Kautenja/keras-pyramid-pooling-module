"""
The Pyramid Pooling Module from PSPNet.

Reference:
    Authors: H. Zhao, J. Shi, X. Qi, X. Wang, J. Jia
    Paper: Pyramid Scene Parsing Network
    URL: https://arxiv.org/pdf/1612.01105.pdf

"""
from keras.engine.topology import Layer
from keras.engine.base_layer import InputSpec
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
import keras.backend as K


class PyramidPoolingModule(Layer):
    """
    The Pyramid Pooling Module from PSPNet.

    Reference:
        Authors: H. Zhao, J. Shi, X. Qi, X. Wang, J. Jia
        Paper: Pyramid Scene Parsing Network
        URL: https://arxiv.org/pdf/1612.01105.pdf

    """

    def __init__(self,
        bin_sizes=[1, 2, 3, 6],
        pool_mode='avg',
        pool_padding='valid',
        conv_padding='valid',
        data_format=None,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        """
        Initialize a new Pyramid Pooling Module.

        Args:
            bin_sizes: sizes for pooling bins
            pool_mode: pooling mode to use
            pool_padding: One of `"valid"` or `"same"` (case-insensitive).
            conv_padding: One of `"valid"` or `"same"` (case-insensitive).
            data_format: one of "channels_last" or "channels_first"
            activation: Activation function to use
            use_bias: whether layer uses a bias vector
            kernel_initializer: Initializer for kernel weights
            bias_initializer: Initializer for bias vector
            kernel_regularizer: Regularizer function applied to kernel weights
            bias_regularizer: Regularizer function applied to bias vector
            activity_regularizer: Regularizer function applied to output
            kernel_constraint: Constraint function applied to kernel
            bias_constraint: Constraint function applied to bias vector
            kwargs: keyword arguments for Layer super constructor

        Returns:
            None

        """
        # setup instance variables
        self.input_spec = InputSpec(ndim=4)
        self.bin_sizes = bin_sizes
        self.pool_mode = pool_mode
        self.pool_padding = conv_utils.normalize_padding(pool_padding)
        self.conv_padding = conv_utils.normalize_padding(conv_padding)
        self.data_format = K.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # initialize the kernels and biases
        self.kernels = None
        self.biases = None
        # call the super constructor
        super(PyramidPoolingModule, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer for the given input shape.

        Args:
            input_shape: the shape to build the layer with

        Returns:
            None

        """
        # determine which axis contains channel data
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        # if the channel dimension is not defined, raise an error
        if input_shape[channel_axis] is None:
            raise ValueError(
                'The channel dimension of the inputs should be defined. '
                'Found `None`. '
            )

        # get the input channels from the input shape
        input_dim = input_shape[channel_axis]
        # get the number of channels for the convolutional kernel as 1 / N
        # of the input dimension (where N is the level size of the pyramid)
        output_dim = input_dim // len(self.bin_sizes)
        # create the shape for the N 1 x 1 kernels
        kernel_shape = (1, 1, input_dim, output_dim)

        # initialize the kernels and biases as empty lists
        self.kernels = len(self.bin_sizes) * [None]
        self.biases = len(self.bin_sizes) * [None]
        # iterate over the levels in the pyramid
        for (level, bin_size) in enumerate(self.bin_sizes):
            # create the kernel weights for this level
            self.kernels[level] = self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                name='kernel_{}'.format(bin_size),
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint
            )
            # if using bias, create the bias weights for this level
            if self.use_bias:
                self.biases[level] = self.add_weight(
                    shape=(output_dim, ),
                    initializer=self.bias_initializer,
                    name='bias_{}'.format(bin_size),
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint
                )

        # set input specification for th layer
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        """
        Return the output shape of the layer for given input shape.

        Args:
            input_shape: the input shape to transform to output shape

        Returns:
            the output shape as a function of input shape

        """
        # calculate the number of output filters used by each pyramid level
        level_filters = (input_shape[-1] // len(self.bin_sizes))
        # calculate the number of filters produced by the pyramid
        pyramid_filters = len(self.bin_sizes) * level_filters
        # concatenate input filters with pyramid filters to determine the
        # number of output filters produced by the module
        output_filters = input_shape[-1] + pyramid_filters
        # use the number of output filters to determine the output shape
        output_shape = input_shape[:-1] + (output_filters, )

        return output_shape

    def call(self, input_):
        """
        Forward pass through the layer.

        Args:
            input_: the input tensor to pass through the pyramid pooling module

        Returns:
            the output tensor from the pyramid pooling module

        """
        # the shape to up-sample pooled tensors to
        output_shape = K.shape(input_)[1:-1]
        # create a list of output tensors to concatenate together
        output_tensors = [input_]
        # iterate over the bin sizes in the pooling module
        for (level, bin_size) in enumerate(self.bin_sizes):
            # pass the inputs through the pooling layer with the given bin
            # size, i.e., a square kernel with side matching the bin size and
            # a matching stride
            x = K.pool2d(input_, (bin_size, bin_size),
                strides=(bin_size, bin_size),
                padding=self.pool_padding,
                pool_mode=self.pool_mode,
            )
            # pass the pooled valued through a 1 x 1 convolution
            x = K.conv2d(x, self.kernels[level],
                strides=(1, 1),
                padding=self.conv_padding,
                data_format=self.data_format,
            )
            # if use bias, apply the bias to the convolved outputs
            if self.use_bias:
                x = K.bias_add(x, self.biases[level],
                    data_format=self.data_format,
                )
            # apply the activation function if there is one
            if self.activation is not None:
                x = self.activation(x)
            # up-sample the outputs back to the input shape
            x = K.tensorflow_backend.tf.image.resize_bilinear(x, output_shape)
            # concatenate the output tensor with the stack of output tensors
            output_tensors += [x]

        return K.concatenate(output_tensors, axis=-1)

    def get_config(self):
        """Return the configuration for building the layer."""
        # generate a dictionary of configuration items for this layer
        config = dict(
            bin_sizes=self.bin_sizes,
            pool_mode=self.pool_mode,
            pool_padding=self.pool_padding,
            conv_padding=self.conv_padding,
            data_format=self.data_format,
            activation=activations.serialize(self.activation),
            use_bias=self.use_bias,
            kernel_initializer=initializers.serialize(self.kernel_initializer),
            bias_initializer=initializers.serialize(self.bias_initializer),
            kernel_regularizer=regularizers.serialize(self.kernel_regularizer),
            bias_regularizer=regularizers.serialize(self.bias_regularizer),
            activity_regularizer=regularizers.serialize(self.activity_regularizer),
            kernel_constraint=constraints.serialize(self.kernel_constraint),
            bias_constraint=constraints.serialize(self.bias_constraint),
        )
        # get the base configuration from the parent constructor
        base_config = super(PyramidPoolingModule, self).get_config()
        # return the dictionary of configuration items for the layer
        return dict(list(base_config.items()) + list(config.items()))


# explicitly define the outward facing API of this module
__all__ = [PyramidPoolingModule.__name__]
