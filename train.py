# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train
   Description :
   Author :       iffly
   date：          5/12/18
-------------------------------------------------
   Change Activity:
                   5/12/18:
-------------------------------------------------
"""
import os
import re

import tensorflow as tf

import dataset_records
from config import Config
from p3d_model import P3D63

BATCH_SIZE = 20
TRAIN_LEN = 212290
FINE_PATH = './best_model_63/c3d_ucf_model_best____'
TRAIN_PATH = './dataset/ucf_data/train_data.tfrecords'
TEST_PATH = './dataset/ucf_data/test_data.tfrecords'
EPOCH = 10
NUM_CLASS = 101
MOVING_AVERAGE_DECAY = 0.9999
MODEL_SAVE_PATH = './models/ucf/p3d/models_63'
BEST_MODEL_SAVE_PATH = './models/ucf/p3d/best_models_63'
TENSORBOARD_PATH = './tensor_logs/ucf/p3d_log_63/'
BEST_FILE = './best.txt'
ACC_FILE = './acc.txt'
LR1 = 1e-4
LR2 = 1e-4
DECAY_RATE1 = 0.5
DECAY_RATE2 = 0.5
STEP_INV1 = 5
STEP_INV2 = int(TRAIN_LEN / BATCH_SIZE)


def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
if not os.path.exists(BEST_MODEL_SAVE_PATH):
    os.makedirs(BEST_MODEL_SAVE_PATH)
with tf.Graph().as_default():
    global_step = tf.get_variable(
        'global_step',
        [],
        initializer=tf.constant_initializer(0),
        trainable=False, dtype=tf.int32
    )

    boundaries1 = [int(TRAIN_LEN / BATCH_SIZE * 5), int(TRAIN_LEN / BATCH_SIZE * 7)]
    values1 = [LR1, LR1 * 0.1, LR1 * 0.01]

    learning_rate1 = tf.train.piecewise_constant(global_step, boundaries1, values1)

    # learning_rate1 = tf.train.exponential_decay(LR1, global_step, decay_steps=TRAIN_LEN/BATCH_SIZE, decay_rate=DECAY_RATE1,
    #                                            staircase=True)
    # learning_rate2 = tf.train.exponential_decay(LR2, global_step1, decay_steps=TRAIN_LEN/BATCH_SIZE, decay_rate=DECAY_RATE2,
    #                                            staircase=True)
    opt_stable = tf.train.AdamOptimizer(learning_rate1)
    images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           Config.channels,
                                                           Config.image_size,
                                                           Config.image_size,
                                                           3))
    labels_placeholder = tf.placeholder(tf.int64, shape=(None))
    keep_prob = tf.placeholder(tf.float32)

    train_data = dataset_records.dataset_records(TRAIN_PATH, BATCH_SIZE, EPOCH)

    p3d = P3D63(num_classes=NUM_CLASS, dropout=keep_prob)
    logit = p3d(images_placeholder)
    print logit
    accuracy = tower_acc(logit, labels_placeholder)

    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logit),
        name='sparse_cross_loss'
    )

    weight_decay_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    weight_decay_loss = tf.add_n(weight_decay_loss, name="w_loss")
    loss = tf.add(cross_entropy_mean, weight_decay_loss, name='t_loss')

    # grads1 = opt_stable.compute_gradients(cross_entropy_mean, varlist1)
    # grads2 = opt_finetuning.compute_gradients(cross_entropy_mean, varlist2)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    apply_gradient_op1 = opt_stable.minimize(loss, global_step=global_step)
    train_op = apply_gradient_op1
    # Create a saver for writing training checkpoints.
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
    saver = tf.train.Saver(var_list=g_list)
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()
    sess.run(init)
    if os.path.isfile(FINE_PATH) or os.path.isfile(FINE_PATH + '.data-00000-of-00001'):
        saver.restore(sess, FINE_PATH)
        print("fine restore")
    # merged = tf.summary.merge_all()

    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    cross_loss_summary = tf.summary.scalar(
        'cross_loss',
        cross_entropy_mean
    )
    weight_decay_loss_summary = tf.summary.scalar('weight_decay_loss', weight_decay_loss)
    total_loss_summary = tf.summary.scalar('total_loss', loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(TENSORBOARD_PATH + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(TENSORBOARD_PATH + 'test', sess.graph)
    train_iterator = train_data.make_one_shot_iterator()
    train_one_element = train_iterator.get_next()

    step = 0
    best_acc = 0
    best_model_index = -1
    file = open(BEST_FILE, "wb+")
    acc_file = open(ACC_FILE, "wb+")
    p3d.is_training(True, sess)
    try:
        while True:

            data = sess.run(train_one_element)

            if (step) % STEP_INV1 == 0:
                print('Training Data Eval:')
                summary, acc, cross_loss_val, _ = sess.run(
                    [merged, accuracy, cross_entropy_mean, train_op],
                    feed_dict={images_placeholder: data[0],
                               labels_placeholder: data[1],
                               keep_prob: 0.6,
                               })
                print ("{0}:train accuracy: {1:.5f}   loss: {2:.5f}".format(step, acc, cross_loss_val))
                # global_step_v,global_step1_v=sess.run([global_step,global_step1])
                # print (global_step_v,global_step1_v)
                train_writer.add_summary(summary, step)
            else:
                sess.run(train_op, feed_dict={
                    images_placeholder: data[0],
                    labels_placeholder: data[1],
                    keep_prob: 0.6,
                })
            if step % STEP_INV2 == 0 and step != 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, 'c3d_ucf_model'), global_step=step)
                test_data = dataset_records.dataset_records(TEST_PATH, BATCH_SIZE, 1, istrain=False)
                test_iterator = test_data.make_one_shot_iterator()
                test_one_element = test_iterator.get_next()
                print('Validation Data Eval:')
                accuracy_total = 0
                c_loss_total = 0
                len_total = 0
                test_step = 0
                try:
                    p3d.is_training(False, sess)
                    while True:
                        test_data = sess.run(test_one_element)
                        acc, c_loss = sess.run(
                            [accuracy, cross_entropy_mean],
                            feed_dict={
                                images_placeholder: test_data[0],
                                labels_placeholder: test_data[1],
                                keep_prob: 1,
                            })
                        len_total += len(test_data[1])
                        c_loss_total += c_loss
                        accuracy_total += acc * len(test_data[1])
                        test_step += 1
                        # test_writer.add_summary(summary, step)

                except tf.errors.OutOfRangeError:

                    acc_avg = accuracy_total / len_total
                    c_loss_avg = c_loss_total / test_step
                    print (
                        "{0}:test accuracy:{1:.5f} loss:{2:.5f}".format(step, acc_avg, c_loss_avg))
                    acc_file.write("{0}:test accuracy:{1:.5f} loss:{2:.5f}\n".format(step, acc_avg, c_loss_avg))
                    test_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=acc_avg)]),
                        step)
                    test_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='cross_loss', simple_value=c_loss_avg)]),
                        step)
                    if acc_avg > best_acc:
                        best_acc = acc_avg
                        best_model_index = step
                        saver.save(sess, os.path.join(BEST_MODEL_SAVE_PATH, 'c3d_ucf_model_best'))
                        print ("best_model is:%d loss:%.5f  acc:%.5f" % (best_model_index, c_loss, acc_avg))
                        file.write("best_model is:%d loss:%.5f  acc:%.5f\n" % (best_model_index, c_loss, acc_avg))
                    p3d.is_training(True, sess)

            step += 1
    except tf.errors.OutOfRangeError:
        print ("best_model is:%d" % (best_model_index))
        print("end!")
