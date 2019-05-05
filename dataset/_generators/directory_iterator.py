"""An iterator capable of reading tensors from a directory on disk."""
import os
import glob
from keras_preprocessing.image import Iterator
import numpy as np
from skimage.transform import resize
from skimage.io import imread


class DirectoryIterator(Iterator):
    """An iterator capable of reading tensors from a directory on disk."""

    def __init__(self,
        directory: str,
        image_data_generator: 'ImageDataGenerator',
        image_size: tuple=(352, 480),
        target_size: tuple=(256, 256),
        batch_size: int=32,
        shuffle: bool=True,
        seed: int=None,
        interpolation: str='nearest',
        data_format='channels_last',
        dtype: str='float32',
        white_list_formats: set={
            'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff', 'npy', 'npz'
        },
    ) -> None:
        """
        Initialize a new directory iterator.

        Args:
            directory: Path to the directory to read images from
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization
            image_size: the size of input images to load
            target_size: the dimensions to resize input images to
            batch_size: Integer, size of a batch
            shuffle: Boolean, whether to shuffle the data between epochs
            seed: Random seed for data shuffling
            interpolation: Interpolation method used to resample the image if
                the target size is different from that of the loaded image
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box"
                and "hamming" are also supported. By default, "nearest" is used
            data_format: Image data format ("channels_first" or "channels_last")
            dtype: Dtype to use for generated arrays
            white_list_formats: file extensions to white list in the iterator

        Returns:
            None

        """
        # store instance variables
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.image_size = image_size
        self.target_size = tuple(target_size)
        self.interpolation = interpolation
        self.data_format = data_format
        self.dtype = dtype
        # iterate over the white-listed formats to load files into the queue
        self.filenames = []
        for format_ in white_list_formats:
            regex = os.path.join(self.directory, '*.{}'.format(format_))
            self.filenames += glob.glob(regex)
        # count the number of filenames as the number of total samples
        self.filenames = sorted(self.filenames)
        self.samples = len(self.filenames)
        # call the super constructor
        super().__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index: list) -> np.ndarray:
        """
        Return a batch of transformed data.

        Args:
            index: the array of filenames to load in the batch

        Returns:
            a NumPy array with the batch of transformed data

        """
        # create an empty list to store batches of images in
        batch_img = [None] * len(index)
        # build batch of image data
        for i, j in enumerate(index):
            # load the image from disk
            if self.image_data_generator.is_numpy:
                img = np.load(self.filenames[j])['y']
            else:
                img = imread(self.filenames[j])
            # resize the image to the correct size
            img = resize(img, self.image_size,
                anti_aliasing=False,
                mode='symmetric',
                clip=False,
                preserve_range=True,
            )
            # apply the transform with the randomized parameters
            img = self.image_data_generator.random_transform(img)
            # standardize the image
            img = self.image_data_generator.standardize(img)
            # permute the dimensions if the data format is channels first
            if self.data_format == 'channels_first':
                img = np.transpose(img, (2, 0, 1))
            # add the image to the batch
            batch_img[i] = img
        # stack the images in the batch list into a vector
        batch_img = np.stack(batch_img).astype(self.dtype)

        return batch_img

    def next(self) -> np.ndarray:
        """Return the next batch of data."""
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be
        # done in parallel
        return self._get_batches_of_transformed_samples(index_array)


# explicitly define the outward facing API of this module
__all__ = [DirectoryIterator.__name__]
