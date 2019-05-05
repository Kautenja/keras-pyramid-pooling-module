"""An object for interacting with RGB semantic segmentation datasets."""
import ast
import os
from keras import backend as K
import numpy as np
import pandas as pd
from ._create_segmented_y import create_segmented_y
from ._generators import ImageDataGenerator
from ._load_label_metadata import load_label_metadata


class DataSet(object):
    """An object for interacting with RGB semantic segmentation datasets."""

    @classmethod
    def load_mapping(cls, mapping_file: str, sep: str=r'\s+') -> dict:
        """
        Load a mapping file from disk as a dictionary.

        Args:
            mapping_file: file pointing to a text file with mapping data
            sep: the separator for entries in the file

        Returns:
            a dictionary mapping old classes to generalized classes

        """
        # the names of the columns in the file
        names = ['og', 'new']
        # load the DataFrame with the original classes as the index col
        mapping = pd.read_table(mapping_file,
            sep=sep,
            names=names,
            index_col='og'
        )
        # return a dict of the new column mapping old classes to new classes
        return mapping['new'].to_dict()

    def __init__(self,
        path_: str,
        mapping: dict=None,
        ignored_labels: list=[],
        target_size: tuple=(720, 960),
        crop_size: tuple=(224, 224),
        horizontal_flip: bool=False,
        vertical_flip: bool=False,
        batch_size: int=3,
        shuffle: bool=True,
        brightness_range: tuple=None,
        zca_epsilon: float=1e-6,
        zca_whitening: bool=False,
        shear_range: float=0.0,
        rotation_range: float=0.0,
        channel_shift_range: float=0.0,
        zoom_range: float=0.0,
        data_format=None,
        seed: int=1,
    ) -> None:
        """
        Initialize a new CamVid dataset instance.

        Args:
            path_: the path to the dataset data
            mapping: mapping to use when generating the preprocessed targets
            ignored_labels: a list of string label names to ignore (0 weight)
            target_size: the image size of the dataset
            crop_size: the size to crop images to. if None, apply no crop
            horizontal_flip: whether to randomly flip images horizontally
            vertical_flip whether to randomly flip images vertically
            batch_size: the number of images to load per batch
            shuffle: whether to shuffle images in the dataset
            brightness_range: a tuple of values for adjusting brightness
            zca_epsilon: the epsilon value to use if zca_whitening is True
            zca_whitening: whether to use ZCA whitening
            shear_range: the max value to shear images
            rotation_range: a max degree measure for randomly rotating images
            channel_shift_range: the max value to randomly shift channels by
            zoom_range: a value or tuple of values denoting random zoom range
            data_format: Image data format ("channels_first" or "channels_last")
            seed: the random seed to use for the generator

        Returns:
            None

        """
        # check type and value of `path_`
        if not isinstance(path_, str):
            raise TypeError('path_ must be a string path')
        # ensure that the base dataset directory exists
        if not os.path.isdir(path_):
            raise ValueError('path_ must point to a dataset directory')
        # ensure that the X directory exists
        if not os.path.isdir(os.path.join(path_, 'X')):
            raise ValueError('path_ directory must contain a directory X/')
        # ensure that the y directory exists
        if not os.path.isdir(os.path.join(path_, 'y')):
            raise ValueError('path_ directory must contain a directory y/')
        # ensure that the label mapping exists
        if not os.path.isfile(os.path.join(path_, 'label_colors.txt')):
            raise ValueError('path_ directory must contain a file label_colors.txt')
        # store the X directory
        self._x = os.path.join(path_, 'X')
        # if the mapping is a string and a path to a file, try to load it
        if isinstance(mapping, str) and os.path.isfile(mapping):
            mapping = self.load_mapping(mapping)
        # load the label metadata from the dataset using the mapping
        metadata = load_label_metadata(path_, mapping)
        # create the segmented dataset and store the path to it
        self._y = create_segmented_y(path_, metadata, mapping)
        # store remaining keyword arguments
        self.ignored_labels = ignored_labels
        self.target_size = target_size
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.batch_size = batch_size
        self.shuffle = shuffle
        if brightness_range is not None:
            if all([not bool(x) for x in brightness_range]):
                brightness_range = None
        self.brightness_range = brightness_range
        self.zca_epsilon = zca_epsilon
        self.zca_whitening = zca_whitening
        self.shear_range = shear_range
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.data_format = K.normalize_data_format(data_format)
        self.seed = seed
        # create a vectorized method to map discrete codes to RGB pixels
        self._unmap = np.vectorize(self.discrete_to_rgb_map.get)
        self._samples = None

    def _data_gen_args(self, context: str) -> dict:
        """
        Return the keyword arguments for creating a new data generator.

        Args:
            context: the context for the call (i.e., 'train' for training)

        Returns:
            a dictionary of keyword arguments to pass to DataGenerator.__init__

        """
        # return for training
        if context == 'train':
            return dict(
                crop_size=self.crop_size,
                horizontal_flip=self.horizontal_flip,
                vertical_flip=self.vertical_flip,
                brightness_range=self.brightness_range,
                zca_epsilon=self.zca_epsilon,
                zca_whitening=self.zca_whitening,
                channel_shift_range=self.channel_shift_range,
                shear_range=self.shear_range,
                rotation_range=self.rotation_range,
                zoom_range=self.zoom_range,
                fill_mode='constant',
                data_format=self.data_format,
            )
        # return for validation / testing (i.e., inference)
        return dict(crop_size=self.crop_size, data_format=self.data_format)

    def _flow_args(self, context: str) -> dict:
        """
        Return the keyword arguments for flowing from a data generator.

        Args:
            context: the context for the call (i.e., 'train' for training)

        Returns:
            a dictionary of keyword arguments to pass to flow_from_directory

        """
        # return for training
        if context == 'train':
            return dict(
                batch_size=self.batch_size,
                target_size=self.target_size,
                shuffle=self.shuffle,
                seed=self.seed
            )
        # return for validation / testing (i.e., inference)
        return dict(
            batch_size=1,
            target_size=self.target_size,
            seed=self.seed
        )

    def _discrete_dict(self, col: str) -> dict:
        """
        Return a dictionary mapping discrete codes to values in another column.

        Args:
            col: the name of the column to map discrete code values to

        Returns:
            a dictionary mapping unique codes to values in the given column

        """
        return self.metadata[['code', col]].set_index('code').to_dict()[col]

    @property
    def discrete_to_rgb_map(self) -> dict:
        """Return a dictionary mapping discrete codes to RGB pixels."""
        rgb_draw = self._discrete_dict('rgb_draw')
        # convert the strings in the RGB draw column to tuples
        return {k: ast.literal_eval(v) for (k, v) in rgb_draw.items()}

    @property
    def discrete_to_label_map(self) -> dict:
        """Return a dictionary mapping discrete codes to string labels."""
        return self._discrete_dict('label_used')

    @property
    def label_to_discrete_map(self) -> dict:
        """Return a dictionary mapping string labels to discrete codes."""
        return {v: k for (k, v) in self.discrete_to_label_map.items()}

    @property
    def ignored_codes(self) -> list:
        """Return a list of the ignored discrete coded labels."""
        # turn the label to discrete code map into a vectorized function
        get = np.vectorize(self.label_to_discrete_map.get, otypes=['uint64'])
        # unwrap the codes for each label in the ignored labels list
        ignored = get(self.ignored_labels)
        # return the ignored codes
        return list(ignored)

    @property
    def codes(self) -> dict:
        """Return the codes of the dataset."""
        return np.array(self._discrete_dict('label_used').keys())

    @property
    def labels(self) -> dict:
        """Return the labels of the dataset."""
        return np.array(self._discrete_dict('label_used').values())

    @property
    def metadata(self) -> pd.DataFrame:
        """Return the metadata associated with this dataset."""
        return pd.read_csv(os.path.join(self._y, 'metadata.csv'))

    @property
    def shape(self):
        """Return the image shape of this data generator."""
        if self.data_format == 'channels_first':
            return (3, *self.crop_size)
        return (*self.crop_size, 3)

    @property
    def num_classes(self) -> int:
        """Return the number of training classes in this dataset."""
        return len(self.metadata['code'].unique())

    @property
    def class_weights(self) -> dict:
        """Return a dictionary of class weights keyed by discrete label."""
        weights = pd.read_csv(os.path.join(self._y, 'weights.csv'), index_col=0)
        # calculate the frequency of each class (and swap to NumPy)
        freq = (weights['pixels'] / weights['pixels_total']).values
        # ignore the ignored values when calculating median
        med = np.median(np.delete(freq, self.ignored_codes))
        # calculate the weights as the median frequency divided by all freq
        weights = (med / freq)
        # set ignored weights to 0
        weights[self.ignored_codes] = 0

        return weights

    @property
    def class_mask(self) -> dict:
        """Return a dictionary of class weights keyed by discrete label."""
        weights = self.class_weights
        # get the class mask as a boolean vector
        class_mask = weights > 0
        # cast the boolean vector to integers for math
        class_mask = class_mask.astype(weights.dtype)

        return class_mask

    @property
    def samples(self) -> dict:
        """Return the number of samples per subset."""
        if self._samples is None:
            self.generators()
        return self._samples

    @property
    def steps_per_epoch(self) -> int:
        """Return the number of steps per training epoch."""
        if 'train' not in self.samples:
            raise ValueError('no training set in this dataset.')
        return int(self.samples['train'] / self.batch_size)

    @property
    def validation_steps(self) -> int:
        """Return the number of validation steps."""
        if 'val' not in self.samples:
            raise ValueError('no validation set in this dataset.')
        return int(self.samples['val'])

    @property
    def test_steps(self) -> int:
        """Return the number of test steps."""
        if 'test' not in self.samples:
            raise ValueError('no testing set in this dataset.')
        return int(self.samples['test'])

    def unmap(self, y_discrete: np.ndarray) -> np.ndarray:
        """
        Un-map a one-hot vector y frame to the target RGB values.

        Args:
            y_discrete: the one-hot vector to convert to an RGB image

        Returns:
            an RGB encoding of the one-hot input tensor

        """
        return np.stack(self._unmap(y_discrete.argmax(axis=-1)), axis=-1)

    def generators(self) -> dict:
        """Return a dictionary with both training and validation generators."""
        # the dictionary to hold generators by key value (training, validation)
        generators = dict()
        self._samples = dict()
        # iterate over the generator subsets
        for subset in next(os.walk(self._y))[1]:
            # create an image generator for loading source images from disk
            x_g = ImageDataGenerator(**self._data_gen_args(subset))
            # create an image generator for loading one-hot tensors from disk
            y_g = ImageDataGenerator(True, **self._data_gen_args(subset))
            # get the path for the subset of data
            _x = os.path.join(self._x, subset)
            _y = os.path.join(self._y, subset)
            # get the directory iterators for the subset
            x_dir = x_g.flow_from_directory(_x, **self._flow_args(subset))
            y_dir = y_g.flow_from_directory(_y, **self._flow_args(subset))
            # combine X and y generators into a single generator with repeats
            generators[subset] = zip(x_dir, y_dir)
            # get the number of samples for this subset
            self._samples[subset] = y_dir.samples

        return generators


# explicitly define the outward facing API of this module
__all__ = [DataSet.__name__]
