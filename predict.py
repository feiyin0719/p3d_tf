# -*- coding: utf-8 -*-
# @Time    : 18-5-18 上午11:42
# @Author  : iffly
# @File    : predict.py
import pickle
import re

import cv2
import numpy as np
import tensorflow as tf

from config import Config
from p3d_model import P3D63

NUM_CLASS = 101
MODEL_PATH = './models/ucf/p3d/best_models_63/c3d_ucf_model_best'
# MODEL_PATH='./sports1m_finetuning_ucf101.model'
VIDEO_PATH = './dataset/UCF-101/JumpRope/v_JumpRope_g06_c04.avi'
# VIDEO_PATH='../p3d_yf/testvideos/wotui1.mp4'
LABEL_PATH = './dataset/ucf_data/label.pickle'
# TEST_PATH='./dataset/ucf_data/train_data.tfrecords'
with tf.Graph().as_default():
    images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           Config.channels,
                                                           Config.image_size,
                                                           Config.image_size,
                                                           3))
    p3d = P3D63(num_classes=NUM_CLASS, dropout=1)
    net = p3d(images_placeholder)
    # model=tl.DNN()
    # tf.layers.batch_normalization
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
    print var_list
    saver = tf.train.Saver(var_list=var_list)
    # print var_list
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    sess.run(init)
    saver.restore(sess, MODEL_PATH)
    # tl.DNN
    # print sess.run(bn_moving_vars)
    # print sess.run(bn_moving_vars)
    p3d.is_training(False, sess)
    # print bn_moving_vars[-1]
    # print sess.run(bn_moving_vars[-1])
    capture = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    with open(LABEL_PATH, 'rb') as f:
        class_name = pickle.load(f)
    print class_name
    np_mean = np.load('crop_mean.npy').reshape([Config.channels, Config.image_size, Config.image_size, 3])
    if capture.isOpened():
        now_frame = 0
        now_index = 0
        while True:
            success, frame = capture.read()

            if not success:
                break

            if now_frame % 8 == 0:
                frame = cv2.resize(frame, (Config.image_size, Config.image_size))
                frame = frame.astype(np.float32)
                frame = np.reshape(frame, (Config.image_size, Config.image_size, 3))

                frame = frame - np_mean[now_index]
                # print sum(frame)
                now_index += 1
                frames.append(frame)
            if len(frames) == 16:
                frames__ = [frames]
                logit = sess.run(net, feed_dict={images_placeholder: frames__})
                frames = []
                now_index = 0
                # print (logit)
                label = np.argmax(logit[0])
                print ("action is:{}({})".format(class_name[label], label))
            now_frame += 1
        capture.release()
    else:
        print("cannot open " + VIDEO_PATH)
    # test_data = dataset_records.dataset_records(TEST_PATH,32, 1, shuffle=False)
    # test_iterator = test_data.make_one_shot_iterator()
    # test_one_element = test_iterator.get_next()
    # try:
    #     i=0
    #     while True:
    #         data = sess.run(test_one_element)
    #         logit=sess.run(net,feed_dict={images_placeholder:data[0]})
    #
    #
    #         label=np.argmax(logit[0])
    #         print ("action is:{}({})".format(class_name[label],label))
    #         i=+1
    #         if i==100:
    #             break
    # except tf.errors.OutOfRangeError:
    #     pass
    # var_list = tf.trainable_variables()
    # g_list = tf.global_variables()
    # pattern1 = re.compile(r"BatchNormalization[^/]*/moving_mean")
    # pattern2 = re.compile(r"BatchNormalization[^/]*/moving_variance")
    # bn_moving_vars = [g for g in g_list if re.search(pattern1, g.name)]
    # bn_moving_vars += [g for g in g_list if re.search(pattern2, g.name)]
    # var_list += bn_moving_vars
    # saver = tf.train.Saver(var_list)
    # saver.save(sess, os.path.join('./best_63', 'c3d_ucf_model_best'))
    # print("end!")
