# Keras Pyramid Pooling Module

[![Build Status][build-status]][ci-server]
[![PackageVersion][pypi-version]][pypi-home]
[![PythonVersion][python-version]][python-home]
[![Stable][pypi-status]][pypi-home]
[![Format][pypi-format]][pypi-home]
[![License][pypi-license]](LICENSE)

[build-status]: https://travis-ci.com/Kautenja/keras-pyramid-pooling-module.svg?branch=master
[ci-server]: https://travis-ci.com/Kautenja/keras-pyramid-pooling-module
[pypi-version]: https://badge.fury.io/py/keras-pyramid-pooling-module.svg
[pypi-license]: https://img.shields.io/pypi/l/keras-pyramid-pooling-module.svg
[pypi-status]: https://img.shields.io/pypi/status/keras-pyramid-pooling-module.svg
[pypi-format]: https://img.shields.io/pypi/format/keras-pyramid-pooling-module.svg
[pypi-home]: https://badge.fury.io/py/keras-pyramid-pooling-module
[python-version]: https://img.shields.io/pypi/pyversions/keras-pyramid-pooling-module.svg
[python-home]: https://python.org

A [Keras](https://keras.io) implementation of the Pyramid Pooling Module
discussed in [_Pyramid scene parsing network [1]_](#references).

## Installation

The preferred installation of `keras-pyramid-pooling-module` is from `pip`:

```shell
pip install keras-pyramid-pooling-module
```

## Usage

The module functions as any other convolutional / pooling layer applied to a
rank 4 tensor (batch, height, width, channels):

```python
from keras.layers import Input
from keras.models import Model
from keras_pyramid_pooling_module import PyramidPoolingModule


input_ = Input((224, 224, 3))
x = PyramidPoolingModule()(input_)
model = Model(inputs=input_, outputs=x)
```

# References

[_[1] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia. Pyramid scene parsing network. CVPR, 2017._][ref1]

[ref1]: https://hszhao.github.io/projects/pspnet/
