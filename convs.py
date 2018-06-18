# -*- coding: utf-8 -*-
# @Time    : 18-5-26 下午2:03
# @Author  : iffly
# @File    : convs.py
import tensorflow as tf
from tensorflow.contrib import layers

from bn import batch_norm


class Conv3D(object):
    def __init__(self, out_planes, filter_size, strides=1, padding='same',
                 weights_init=tf.variance_scaling_initializer(), bias=False, activation=None,
                 w_regularizer=layers.l2_regularizer(0.00005), trainable=True, name='conv_3d'):
        self.out_planes = out_planes
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.weights_init = weights_init
        self.name = name
        self.bias = bias
        self.activation = activation
        self.w_regularizer = w_regularizer
        self.trainable = trainable

    def __call__(self, incoming):
        conv = tf.layers.conv3d(incoming, filters=self.out_planes, kernel_size=self.filter_size, strides=self.strides,
                                padding=self.padding,
                                kernel_initializer=self.weights_init, activation=self.activation, use_bias=self.bias,
                                kernel_regularizer=self.w_regularizer, trainable=self.trainable, name=self.name,
                                reuse=False)
        return conv


class Conv2D(object):
    def __init__(self, out_planes, filter_size, strides=1, padding='same',
                 weights_init=tf.variance_scaling_initializer(), bias=False, activation=None,
                 w_regularizer=layers.l2_regularizer(0.00005), trainable=True, name='conv_2d'):
        self.out_planes = out_planes
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.weights_init = weights_init
        self.name = name
        self.bias = bias
        self.activation = activation
        self.w_regularizer = w_regularizer
        self.trainable = trainable

    def __call__(self, incoming):
        conv = tf.layers.conv2d(incoming, filters=self.out_planes, kernel_size=self.filter_size, strides=self.strides,
                                padding=self.padding,
                                kernel_initializer=self.weights_init, activation=self.activation, use_bias=self.bias,
                                kernel_regularizer=self.w_regularizer, trainable=self.trainable, name=self.name,
                                reuse=False)
        return conv


class MaxPool3D(object):
    def __init__(self, kernel_size, strides=1, padding='same', name='maxpool_3d'):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.name = name

    def __call__(self, incoming):
        return tf.layers.max_pooling3d(incoming, pool_size=self.kernel_size, strides=self.strides,
                                       padding=self.padding, name=self.name)


class MaxPool2D(object):
    def __init__(self, kernel_size, strides=1, padding='same', name='maxpool_2d'):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.name = name

    def __call__(self, incoming):
        return tf.layers.max_pooling2d(incoming, pool_size=self.kernel_size, strides=self.strides,
                                       padding=self.padding, name=self.name)


class AvgPool3D(object):
    def __init__(self, kernel_size, strides=1, padding='same', name='avgpool_3d'):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.name = name

    def __call__(self, incoming):
        return tf.layers.average_pooling3d(incoming, pool_size=self.kernel_size, strides=self.strides,
                                           padding=self.padding, name=self.name)


class AvgPool2D(object):
    def __init__(self, kernel_size, strides=1, padding='same', name='avgpool_2d'):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.name = name

    def __call__(self, incoming):
        return tf.layers.average_pooling2d(incoming, pool_size=self.kernel_size, strides=self.strides,
                                           padding=self.padding, name=self.name)


class BatchNormal(object):
    def __init__(self, trainable=True, name='batch_normal', training=False):
        self.name = name
        self.trainable = trainable
        self.training = training

    def __call__(self, incoming):
        return batch_norm(incoming, self.training, trainable=self.trainable, name=self.name)


class Sequential(object):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, incoming):
        net = incoming
        for layer in self.layers:
            net = layer(net)
        return net
