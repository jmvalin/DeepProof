# -*- coding: utf-8 -*-
"""AttnApply Layer
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Layer
from keras import backend as K


class AttnApply(Layer):
    def __init__(self, T, padding=False, **kwargs):
        super(AttnApply, self).__init__(**kwargs)
        self.T = T
        self.padding = padding
        #self.input_spec = InputSpec(ndim=2)

    #def compute_output_shape(self, input_shape):
    #    return (input_shape[0], input_shape[1], self.T*input_shape[2])

    def call(self, input_list):
        inputs, weights = input_list
        D = self.T//2
        if self.padding:
            inputs = K.concatenate([0*inputs[:, :D, :], inputs, 0*inputs[:, -D:, :]], axis=1)
        output = inputs[:, :-self.T+1, :]*weights[:, :, 0:1]
        for i in range(1, self.T-1):
            output = output + inputs[:, i:-self.T+i+1, :]*weights[:, :, i:i+1]
        output = output + inputs[:, self.T-1:, :]*weights[:, :, -1:]
        return output

    def get_config(self):
        config = {'T': self.T}
        base_config = super(AttnApply, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
