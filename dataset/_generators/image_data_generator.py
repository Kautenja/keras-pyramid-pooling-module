"""An image generator for streaming NumPy data from disk."""
import numpy as np
from keras import backend as K
from keras_preprocessing.image import apply_affine_transform
from keras_preprocessing.image import apply_brightness_shift
from keras_preprocessing.image import apply_channel_shift
from keras_preprocessing.image import flip_axis
from .directory_iterator import DirectoryIterator


def crop_dim(dim: int, crop_size: int) -> tuple:
    """
    Return the crop bounds of a dimension using RNG.

    Args:
        dim: the value of the dimension
        crop_size: the value to crop the dimension to

    Returns:
        a tuple of:
        -   the starting point of the crop
        -   the stopping point of the crop

    """
    # if crop size is equal to input size, return the input dimension
    if crop_size == dim:
        return 0, dim
    # otherwise generate a random anchor point and add the crop size to it
    dim_0 = np.random.randint(0, dim - crop_size)
    dim_1 = dim_0 + crop_size

    return dim_0, dim_1


class ImageDataGenerator(object):
    """An image data generator that applies augmentations at runtime."""

    def __init__(self,
        is_numpy: bool=False,
        crop_size: tuple=None,
        rotation_range: int=0,
        brightness_range: tuple=None,
        channel_shift_range: float=0.0,
        shear_range: float=0.0,
        zoom_range: float=0.0,
        fill_mode: str='nearest',
        cval: float=0.0,
        horizontal_flip: bool=False,
        vertical_flip: bool=False,
        rescale: float=None,
        data_format='channels_last',
        dtype='float32',
        **kwargs
    ) -> None:
        """
        Create a new image data generator.

        Args:
            is_numpy: whether the data is stored in NumPy compressed format
            crop_size: a tuple of (height, width) for selecting random crops
            rotation_range: Int. Degree range for random rotations.
            brightness_range: Tuple or list of two floats. Range for picking
                a brightness shift value from.
            channel_shift_range:
            shear_range: Float. Shear Intensity
                (Shear angle in counter-clockwise direction in degrees)
            zoom_range: Float or [lower, upper]. Range for random zoom.
                If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
            fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
                Default is 'nearest'.
                Points outside the boundaries of the input are filled
                according to the given mode:
                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd
            cval: Float or Int.
                Value used for points outside the boundaries
                when `fill_mode = "constant"`.
            horizontal_flip: Boolean. Randomly flip inputs horizontally.
            vertical_flip: Boolean. Randomly flip inputs vertically.
            rescale: rescaling factor. Defaults to None.
                If None or 0, no rescaling is applied,
                otherwise we multiply the data by the value provided
                (after applying all other transformations).
            data_format: Image data format ("channels_first" or "channels_last")
            dtype: Dtype to use for the generated arrays.

        Returns:
            None

        """
        self.is_numpy = is_numpy
        self.crop_size = crop_size
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.channel_shift_range = channel_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.data_format = K.normalize_data_format(data_format)
        self.dtype = dtype

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (zoom_range,))

    def flow_from_directory(self, directory: str,
        target_size: tuple=(256, 256),
        batch_size: int=32,
        shuffle: bool=True,
        seed: int=None,
        interpolation: str='nearest'
    ) -> DirectoryIterator:
        """
        Generate batches of augmented data using a path to a directory.

        Args:
            directory: Path to the target directory.
                It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM or TIF images
                inside each of the subdirectories directory tree
                will be included in the generator.
                See [this script](
                https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
                for more details.
            target_size: Tuple of integers `(height, width)`, to resize images
                to when loading from disk
            batch_size: Size of the batches of data (default: 32).
            shuffle: Whether to shuffle the data (default: True)
            seed: Optional random seed for shuffling and transformations.
            interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`,
                and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed,
                `"box"` and `"hamming"` are also supported.
                By default, `"nearest"` is used.

        Returns:
            A `DirectoryIterator` yielding tuples of `(x, y)`
                where `x` is a numpy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a numpy array of corresponding labels.

        """
        return DirectoryIterator(directory, self,
            image_size=target_size,
            target_size=target_size if self.crop_size is None else self.crop_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            interpolation=interpolation,
            data_format=self.data_format,
            dtype=self.dtype,
        )

    def standardize(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the normalization configuration to a batch of inputs.

        Args:
            x: Batch of inputs to be normalized.

        Returns:
            The inputs, normalized.

        """
        # if rescale is enabled, apply the rescaling
        if self.rescale:
            x *= self.rescale

        return x

    def get_random_transform(self, img_shape, seed: int=None) -> dict:
        """
        Generate random parameters for a transformation.

        Args:
            seed: Random seed.
            img_shape: Tuple of integers.
                Shape of the image that is transformed.

        Returns:
            A dict with randomized parameters describing the transformation.

        """
        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range:
            theta = np.random.uniform(
                -self.rotation_range,
                self.rotation_range)
        else:
            theta = 0

        tx = 0
        ty = 0

        if self.shear_range:
            shear = np.random.uniform(
                -self.shear_range,
                self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        channel_shift_intensity = None
        if self.channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self.channel_shift_range, self.channel_shift_range)

        brightness = None
        if self.brightness_range is not None:
            brightness = np.random.uniform(self.brightness_range[0], self.brightness_range[1])

        crop_size = None
        if self.crop_size is not None:
            crop_row = crop_dim(img_shape[0], self.crop_size[0])
            crop_col = crop_dim(img_shape[1], self.crop_size[1])
            crop_size = crop_row, crop_col

        return dict(
            theta=theta,
            tx=tx,
            ty=ty,
            shear=shear,
            zx=zx,
            zy=zy,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
            channel_shift_intensity=channel_shift_intensity,
            brightness=brightness,
            crop_size=crop_size,
        )

    def apply_transform(self, x: np.ndarray, transform_parameters: dict):
        """
        Apply a transformation to an image according to given parameters.

        Args:
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intencity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.

        Returns:
            A transformed version of the input (same shape).

        """
        x = apply_affine_transform(x,
            transform_parameters.get('theta', 0),
            transform_parameters.get('tx', 0),
            transform_parameters.get('ty', 0),
            transform_parameters.get('shear', 0),
            transform_parameters.get('zx', 1),
            transform_parameters.get('zy', 1),
            row_axis=0,
            col_axis=1,
            channel_axis=2,
            fill_mode=self.fill_mode,
            cval=0
        )

        # TODO: parameterize better (this assumes the last index is Void).
        # this was confusing to figure out when forgotten about
        #
        # if the fill mode is constant, find all the empty vectors and reset
        # them to the Null vector (i.e., the last index is 1)
        if self.fill_mode == 'constant':
            x[x.sum(axis=-1) == 0, -1] = 1

        if transform_parameters.get('channel_shift_intensity') is not None:
            x = apply_channel_shift(x,
                transform_parameters['channel_shift_intensity'],
                channel_axis=2
            )

        if transform_parameters.get('flip_horizontal', False):
            x = flip_axis(x, 1)

        if transform_parameters.get('flip_vertical', False):
            x = flip_axis(x, 0)

        if not self.is_numpy:
            if transform_parameters.get('brightness') is not None:
                x = apply_brightness_shift(x, transform_parameters['brightness'])

        if transform_parameters.get('crop_size') is not None:
            # get the crop dimensions
            crop_h, crop_w = transform_parameters['crop_size']
            x = x[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1], :]

        return x

    def random_transform(self, x: np.ndarray, seed: int=None) -> np.ndarray:
        """
        Apply a random transformation to an image.

        Args:
            x: 3D tensor, single image.
            seed: Random seed.

        Returns:
            A randomly transformed version of the input (same shape).

        """
        params = self.get_random_transform(x.shape, seed)
        return self.apply_transform(x, params)


# explicitly define the outward facing API of this module
__all__ = [ImageDataGenerator.__name__]
