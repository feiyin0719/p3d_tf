# -*- coding: utf-8 -*-
# @Time    : 18-6-1 下午2:33
# @Author  : iffly
# @File    : utils.py
import numpy as np
import tensorflow as tf


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, np.ndarray, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")
