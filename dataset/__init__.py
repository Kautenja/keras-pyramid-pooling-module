"""A generalization of a semantic segmentation DataSet."""
from ._dataset import DataSet


# explicitly define the outward facing API of this module
__all__ = [DataSet.__name__]
