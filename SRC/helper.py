# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:33:35 2020

@author: Mauricio
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types
# Dependency imports
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.distributions import deterministic as deterministic_lib
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow.python.keras.utils import generic_utils
"""
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib



def GetOddsRatio(reference_coeff, case_coeff):
    "Obtiene el 'Odds Ratio (OR) o Razon de Momios dados los coeficientes del termino de bias y de la caracteristica a explorar"
    ref_p = 1/(1 + np.exp(-reference_coeff))
    case_p = 1/(1 + np.exp(-(reference_coeff + case_coeff)))
    RR = case_p/ref_p
    RC = (1 - case_p)/(1 - ref_p)
    RD = case_p - ref_p
    OR = RR/RC
    return OR, RR, RC, RD


def custom_multivariate_normal_fn(loc = None, scale = 1):
    def inner(dtype, shape, name, trainable,
                                   add_variable_fn, loc = loc, scale = scale):
        """Creates multivariate standard `Normal` distribution.
        Args:
            dtype: Type of parameter's event.
            shape: Python `list`-like representing the parameter's event shape.
            name: Python `str` name prepended to any created (or existing)
            `tf.Variable`s.
            trainable: Python `bool` indicating all created `tf.Variable`s should be
            added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
            add_variable_fn: `tf.get_variable`-like `callable` used to create (or
                                                                               access existing) `tf.Variable`s.
                                                                               Returns:
                                                                                   Multivariate standard `Normal` distribution.
    """
        del name, trainable, add_variable_fn   # unused
        if loc is None:
            loc = tf.zeros(shape, dtype)
        dist = normal_lib.Normal(
            loc=loc, scale=dtype.as_numpy_dtype(scale))
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return independent_lib.Independent(
                dist, reinterpreted_batch_ndims=batch_ndims)
    return inner

def logreg_cost_accuracy(X, y, w):
    X, y, w = X.astype('float32'), y.astype('float32'), w.astype('float32')
    h = sigmoid(X @ w)
    cost = ((-y).T @ np.log(h))-((1 -y).T @ np.log(1 - h))
    acc = sum(abs(y - (h < 0.5)))/len(y)
    return cost[0][0], acc[0]

def sigmoid(x):
    return 1/(1+np.exp(-x))
  