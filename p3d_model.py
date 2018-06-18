# -*- coding: utf-8 -*-
# @Time    : 18-5-26 下午1:19
# @Author  : iffly
# @File    : p3d_model.py

import re

import tensorflow as tf
from tensorflow.contrib import layers

from config import Config
from convs import Conv3D, Conv2D, MaxPool3D, AvgPool2D, Sequential, BatchNormal

MODEL_PATH = './best_model_63/c3d_ucf_model_best'


def conv_s(out_planes, strides, padding='same', weights_init=tf.variance_scaling_initializer(), name='conv_s'):
    return Conv3D(out_planes=out_planes, filter_size=[3, 3, 1], strides=strides, padding=padding,
                  weights_init=weights_init, activation=None, bias=False, name=name)


def conv_t(out_planes, strides, padding='same', weights_init=tf.variance_scaling_initializer(), name='conv_t'):
    return Conv3D(out_planes=out_planes, filter_size=[1, 1, 3], strides=strides, padding=padding,
                  weights_init=weights_init, activation=None, bias=False, name=name)


class Bottleneck(object):
    expansion = 4

    def __init__(self, planes, stride=1, downsample=None, n_s=0, depth_3d=47, ST_struc=('A', 'B', 'C'), training=False,
                 name='Bottleneck'):
        self.downsample = downsample
        self.depth_3d = depth_3d
        self.ST_struc = ST_struc
        self.len_ST = len(self.ST_struc)
        # self.batch_normal_index=0
        self.bn = BatchNormal(name='BatchNormalization', training=training)
        self.bn_1 = BatchNormal(name='BatchNormalization_1', training=training)
        self.bn_2 = BatchNormal(name='BatchNormalization_2', training=training)
        self.bn_3 = BatchNormal(name='BatchNormalization_3', training=training)
        self.name = name
        stride_p = stride
        if not self.downsample == None:
            stride_p = [2, 2, 1]
        if n_s < self.depth_3d:
            if n_s == 0:
                stride_p = 1
            self.conv1 = Conv3D(planes, filter_size=1, strides=stride_p, name='conv_3d')
        else:
            if n_s == self.depth_3d:
                stride_p = 2
            else:
                stride_p = 1
            self.conv1 = Conv2D(planes, filter_size=1, bias=False, strides=stride_p, name='conv_2d')
        # self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.id = n_s
        self.ST = list(self.ST_struc)[self.id % self.len_ST]
        if self.id < self.depth_3d:
            self.conv2 = conv_s(planes, strides=1, padding='same')
            #
            self.conv3 = conv_t(planes, strides=1, padding='same')
        else:
            self.conv_normal = Conv2D(planes, filter_size=3, strides=1, padding='same', bias=False, name='conv_normal')
        if n_s < self.depth_3d:

            name = 'conv_3d_1'
            self.conv4 = Conv3D(planes * 4, filter_size=1, bias=False, name=name)

        else:

            name = 'conv_2d_1'
            self.conv4 = Conv2D(planes * 4, filter_size=1, bias=False, name=name)
        self.relu = tf.nn.relu

        self.stride = stride

    # def get_batch_normal(self):
    #     if self.batch_normal_index==0:
    #         self.batch_normal_index+=1
    #         return BatchNormal(name='BatchNormalization')
    #     else:
    #         self.batch_normal_index += 1
    #         return BatchNormal(name='BatchNormalization_{}'.format(self.batch_normal_index))

    def ST_A(self, x):
        x = self.conv2(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn_2(x)
        x = self.relu(x)

        return x

    def ST_B(self, x):
        tmp_x = self.conv2(x)
        tmp_x = self.bn_1(tmp_x)
        tmp_x = self.relu(tmp_x)

        x = self.conv3(x)
        x = self.bn_2(x)
        x = self.relu(x)

        return x + tmp_x

    def ST_C(self, x):
        x = self.conv2(x)
        x = self.bn_1(x)
        x = self.relu(x)

        tmp_x = self.conv3(x)
        tmp_x = self.bn_2(tmp_x)
        tmp_x = self.relu(tmp_x)

        return x + tmp_x

    def __call__(self, x):
        residual = x
        with tf.variable_scope(self.name) as scope:
            out = self.conv1(x)
            out = self.bn(out)
            out = self.relu(out)

            # out = self.conv2(out)
            # out = self.bn2(out)
            # out = self.relu(out)
            if self.id < self.depth_3d:  # C3D parts:

                if self.ST == 'A':
                    out = self.ST_A(out)
                elif self.ST == 'B':
                    out = self.ST_B(out)
                elif self.ST == 'C':
                    out = self.ST_C(out)
            else:
                out = self.conv_normal(out)  # normal is res5 part, C2D all.
                out = self.bn_1(out)
                out = self.relu(out)

            out = self.conv4(out)

            out = self.bn_3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

        return out


class P3D(object):
    def __init__(self, block, layers, modality='RGB',
                 shortcut_type='B', num_classes=400, dropout=0.5, ST_struc=('A', 'B', 'C')):
        self.inplanes = 64
        super(P3D, self).__init__()
        self.training = tf.get_variable('training', [], dtype=tf.bool, trainable=False,
                                        initializer=tf.constant_initializer(False))
        training = self.training
        self.updatetraining = [tf.assign(training, False), tf.assign(training, True)]
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
        #                        padding=(3, 3, 3), bias=False)
        self.input_channel = 3 if modality == 'RGB' else 2  # 2 is for flow
        self.ST_struc = ST_struc
        self.dropout = dropout
        self.num_classes = num_classes
        self.conv1_custom = Conv3D(64, filter_size=[1, 7, 7], strides=[1, 2, 2],
                                   padding='same')

        self.depth_3d = sum(layers[:3])  # C3D layers are only (res2,res3,res4),  res5 is C2D

        self.bn = BatchNormal(name='BatchNormalization', training=self.training)  # bn1 is followed by conv1
        self.cnt = 0
        self.relu = tf.nn.relu
        self.fc = tf.layers.dense
        self.maxpool = MaxPool3D(kernel_size=[2, 3, 3], strides=2, padding='same')  # pooling layer for conv1.
        self.maxpool_2 = MaxPool3D(kernel_size=[1, 1, 2], padding='valid',
                                   strides=[2, 1, 1])  # pooling layer for res2, 3, 4.

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, training=training)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2, training=training)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2, training=training)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2, training=training)

        self.avgpool = AvgPool2D(kernel_size=[5, 5], strides=1)  # pooling layer for res5.

    @property
    def scale_size(self):
        return self.input_size[2] * 256 // 160  # asume that raw images are resized (340,256).

    @property
    def temporal_length(self):
        return self.input_size[1]

    @property
    def crop_size(self):
        return self.input_size[2]

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, training=False):
        downsample = None
        stride_p = stride  # especially for downsample branch.

        if self.cnt < self.depth_3d:
            if self.cnt == 0:
                stride_p = 1
            else:
                stride_p = [2, 2, 1]
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = Sequential(
                    [Conv3D(planes * block.expansion,
                            filter_size=1, strides=stride_p, name='conv_down'),
                     BatchNormal(name='batch_down', training=training), ]
                )


        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = Sequential(
                    [Conv2D(planes * block.expansion,
                            filter_size=1, strides=2, name='conv_down'),
                     BatchNormal(name='batch_down', training=training)
                     ]
                )
        layers = []
        layers.append(block(planes, stride, downsample, n_s=self.cnt, depth_3d=self.depth_3d, ST_struc=self.ST_struc,
                            name='Bottleneck0', training=training))
        self.cnt += 1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes, n_s=self.cnt, depth_3d=self.depth_3d, ST_struc=self.ST_struc,
                                name='Bottleneck{}'.format(i), training=training))
            self.cnt += 1

        return Sequential(layers)

    def is_training(self, training, sess):
        if training:
            sess.run(self.updatetraining[1])
        else:
            sess.run(self.updatetraining[0])

    def __call__(self, x):
        with tf.variable_scope('custom') as scope:
            x = self.conv1_custom(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.maxpool(x)
        with tf.variable_scope('layer1') as scope:
            x = self.maxpool_2(self.layer1(x))  # Part Res2
        with tf.variable_scope('layer2') as scope:
            x = self.maxpool_2(self.layer2(x))  # Part Res3
        with tf.variable_scope('layer3') as scope:
            x = self.maxpool_2(self.layer3(x))  # Part Res4
        with tf.variable_scope('layer4') as scope:
            x = tf.reshape(x, [-1, x.shape[2], x.shape[3], x.shape[4]])
            x = self.layer4(x)
            x = self.avgpool(x)
        with tf.variable_scope('full') as scope:
            x = tf.layers.dropout(x, self.dropout)
            x = tf.reshape(x, [-1, x.shape[3] * x.shape[1] * x.shape[2]])
            x = self.fc(inputs=x, units=self.num_classes, kernel_initializer=tf.variance_scaling_initializer(),
                        kernel_regularizer=layers.l2_regularizer(0.00005), name='FullyConnected')
        return x


def P3D63(**kwargs):
    """Construct a P3D63 modelbased on a ResNet-50-3D model.
    """
    model = P3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def P3D131(**kwargs):
    """Construct a P3D131 model based on a ResNet-101-3D model.
    """
    model = P3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def P3D199(modality='RGB', **kwargs):
    """construct a P3D199 model based on a ResNet-152-3D model.
    """
    model = P3D(Bottleneck, [3, 8, 36, 3], modality=modality, **kwargs)
    return model


if __name__ == '__main__':
    model = P3D63()
    images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           Config.channels,
                                                           Config.image_size,
                                                           Config.image_size,
                                                           3))
    net = model(images_placeholder)
    print net
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # print tensor_util.constant_value(training)
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    pattern1 = re.compile(r"BatchNormalization[^/]*/moving_mean")
    pattern2 = re.compile(r"BatchNormalization[^/]*/moving_variance")
    pattern3 = re.compile(r"batch_down[^/]*/moving_mean")
    pattern4 = re.compile(r"batch_down[^/]*/moving_variance")
    bn_moving_vars = [g for g in g_list if re.search(pattern1, g.name)]
    bn_moving_vars += [g for g in g_list if re.search(pattern2, g.name)]
    bn_moving_vars += [g for g in g_list if re.search(pattern3, g.name)]
    bn_moving_vars += [g for g in g_list if re.search(pattern4, g.name)]
    var_list += bn_moving_vars
    list_temp = set(g_list) - set(var_list)
    print list_temp
