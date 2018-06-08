from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import activations, initializers, regularizers, constraints, InputSpec
import numpy as np

class MultiHead(Layer):
    """Just your regular densely-connected NN layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MultiHead, self).__init__(**kwargs)
        self.units = units
        self.heads = 8
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.input_spec = [InputSpec(min_ndim=3), InputSpec(min_ndim=3), InputSpec(min_ndim=3)]

    def build(self, input_shape):
        self.heads = input_shape[2][-1]//self.units
        assert len(input_shape) >= 2
        query_dim = input_shape[0][-1]
        key_dim = input_shape[1][-1]
        value_dim = input_shape[2][-1]

        self.query_kernel = self.add_weight(shape=(self.heads, query_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='query_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.key_kernel = self.add_weight(shape=(self.heads, key_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='key_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.value_kernel = self.add_weight(shape=(self.heads, key_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='value_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.query_bias = self.add_weight(shape=(self.heads, self.units),
                                        initializer=self.bias_initializer,
                                        name='query_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.key_bias = self.add_weight(shape=(self.heads, self.units),
                                        initializer=self.bias_initializer,
                                        name='key_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.query_bias = None
            self.key_bias = None
        super(MultiHead, self).build(input_shape)

    def call(self, inputs):
        queries, keys, values = inputs
        out_list = []
        for i in range(self.heads):
            q = K.dot(queries, self.query_kernel[i,:,:])
            k = K.dot(keys, self.key_kernel[i,:,:])
            if self.use_bias:
                q = K.bias_add(q, self.query_bias[i,:])
                k = K.bias_add(k, self.key_bias[i,:])
            if self.activation is not None:
                q = self.activation(q)
            weights = K.softmax(K.batch_dot(q, k, axes=[2,2]))
            val = K.dot(values, self.value_kernel[i,:,:])
            out = K.batch_dot(weights, val)
            out_list.append(out)
        output = K.concatenate(out_list, axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape[0])
        output_shape[-1] = input_shape[2][-1]
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MultiHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

