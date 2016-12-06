# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

from nn.utils.theano_utils import *
import nn.initializations as initializations
import nn.activations as activations

from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams


class Layer(object):
    def __init__(self):
        self.params = []

    def __call__(self, X):
        return X

    def get_params(self):
        return self.params

    def set_name(self, name):
        if name:
            for i in range(len(self.params)):
                if self.params[i].name is None:
                    self.params[i].name = '%s_p%d' % (name, i)
                else:
                    self.params[i].name = name + '_' + self.params[i].name

        self.name = name


class Dense(Layer):
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='tanh', name='Dense'):

        super(Dense, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.W.name = '%s_W' % name
        self.b.name = '%s_b' % name

    def __call__(self, X):
        output = self.activation(T.dot(X, self.W) + self.b)
        return output


class Dropout(Layer):
    def __init__(self, p, srng, name='dropout'):
        super(Dropout, self).__init__()

        assert 0. < p < 1.

        self.p = p
        self.srng = srng

        if name is not None:
            self.set_name(name)

    def __call__(self, X, train_only=True):
        retain_prob = 1. - self.p

        X_train = X * self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X_test = X * retain_prob

        if train_only:
            return X_train
        else:
            return X_train, X_test
