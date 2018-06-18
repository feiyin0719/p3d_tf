# -*- coding: utf-8 -*-
# @Time    : 18-5-31 下午4:51
# @Author  : iffly
# @File    : bn.py
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

from utils import get_incoming_shape


def batch_norm(x, training, trainable=True, eps=1e-05, decay=0.9, beta=0.0, gamma=1.0, stddev=0.002, affine=True,
               name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        input_shape = get_incoming_shape(x)
        params_shape = input_shape[-1:]
        input_ndim = len(input_shape)
        axis = list(range(input_ndim - 1))
        gamma_init = tf.random_normal_initializer(mean=gamma, stddev=stddev)
        moving_mean = tf.get_variable('moving_mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, axis, name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay, zero_debias=False),
                                          assign_moving_average(moving_variance, variance, decay, zero_debias=False)]):
                return tf.identity(mean), tf.identity(variance)

        mean, variance = tf.cond(training, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.constant_initializer(beta), trainable=trainable)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=gamma_init, trainable=trainable)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x
