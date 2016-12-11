# -*- coding: utf-8 -*-

from .core import Layer
from nn.utils.theano_utils import *
import nn.initializations as initializations

import nn.activations as activations
from theano.ifelse import ifelse

class Embedding(Layer):
    '''
        Turn positive integers (indexes) into denses vectors of fixed size.
        eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    '''
    def __init__(self, input_dim, output_dim, init='uniform', name=None):

        super(Embedding, self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = self.init((self.input_dim, self.output_dim))

        self.params = [self.W]

        if name is not None:
            self.set_name(name)

    def get_output_mask(self, X):
        return T.ones_like(X, dtype=theano.config.floatX) * (1. - T.eq(X, 0))

    def __call__(self, X, mask_zero=False):
        out = self.W[X]
        if mask_zero:
            return out, self.get_output_mask(X)
        else:
            return out
