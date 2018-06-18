# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     create_tfrecords
   Description :
   Author :       iffly
   date：          5/8/18
-------------------------------------------------
   Change Activity:
                   5/8/18:
-------------------------------------------------
"""
import argparse
import cPickle as pickle

import cv2
import numpy as np
import tensorflow as tf

from config import Config

parser = argparse.ArgumentParser(usage="python create_tfrecords.py --picklepath picklepath --savepath savepath",
                                 description="help info.")
parser.add_argument("--picklepath", default="", help="the pickle path.", dest="picklepath", required=True)
parser.add_argument("--savepath", default="", help="the save path.", dest="savepath", required=True)
args = parser.parse_args()


def create_records(picklepath, savepath):
    with open(picklepath, 'rb') as f:
        data = pickle.load(f)
    writer = tf.python_io.TFRecordWriter(savepath)
    for i in range(len(data)):
        imagepaths = data[i]['data']
        label = data[i]['label']
        images = []
        for imagepath in imagepaths:
            image = cv2.imread(imagepath)
            image = cv2.resize(image, (Config.image_w, Config.image_h))
            images.append(image)
        images = np.array(images)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images.tostring()]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    picklepath = args.picklepath
    savepath = args.savepath
    create_records(picklepath, savepath)
