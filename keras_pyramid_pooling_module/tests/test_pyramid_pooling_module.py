"""Test Cases for the pyramid_pooling_module module."""
from unittest import TestCase
from keras.layers import Input
from keras.layers import Conv2D
from keras.models import Model
from keras import backend as K
import numpy as np
from ..pyramid_pooling_module import PyramidPoolingModule


class ShouldCreatePyramidPoolingModule(TestCase):
    def test(self):
        PyramidPoolingModule()


class ShouldCreatePyramidPoolingModuleWithCorrectSize(TestCase):
    def test(self):
        input_ = Input((224, 224, 12))
        x = PyramidPoolingModule()(input_)
        self.assertEqual((224, 224, 24), K.int_shape(x)[1:])


class ShouldPredict(TestCase):
    def test(self):
        input_ = Input((224, 224, 12))
        x = PyramidPoolingModule()(input_)
        model = Model(inputs=input_, outputs=x)
        y = model.predict(np.random.random((224, 224, 12))[None, ...])
        self.assertEqual((224, 224, 24), y[0].shape)


class ShouldFit(TestCase):
    def test(self):
        input_ = Input((224, 224, 12))
        x = PyramidPoolingModule()(input_)
        x = Conv2D(8, (1, 1), activation='relu')(x)
        model = Model(inputs=input_, outputs=x)
        model.compile(loss='mse', optimizer='adam')
        data = np.random.random((10, 224, 224, 12))
        targets = np.random.random((10, 224, 224, 8))
        model.fit(data, targets, verbose=0)
