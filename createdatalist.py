# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     createdatalist
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
import os
import random

parser = argparse.ArgumentParser(
    usage="python createdatalist.py --imagedir imagedir --labelpath labelpath --trainpath trainpath --testpath testpath [--frac frac]",
    description="help info.")
parser.add_argument("--imagedir", default="", help="the image dir path.", dest="imagedir", required=True)
parser.add_argument("--labelpath", default="", help="the label file save path.", dest="labelpath", required=True)
parser.add_argument("--trainpath", default="", help="the train file save path.", dest="trainpath", required=True)
parser.add_argument("--testpath", default="", help="the test file save path.", dest="testpath", required=True)
parser.add_argument("--frac", default=0.8, help="the test data ratio.", dest="frac")
args = parser.parse_args()


def createdatalist(imagedir, labelpath, trainpath, testpath, frac=0.8):
    class_list = os.listdir(imagedir)
    label_class = {}
    train_data = []
    test_data = []
    label_count = 0
    for class_name in class_list:
        label_class.update({label_count: class_name})
        file_list = os.listdir(os.path.join(imagedir, class_name))
        trainlen = int(len(file_list) * frac)
        print trainlen, len(file_list)
        for i in range(0, trainlen):
            file_name = file_list[i]
            images_path = os.listdir(os.path.join(imagedir, class_name, file_name))
            images_len = len(images_path)
            frame_dis = images_len / 16
            if frame_dis >= 4:
                frame_dis_dis = frame_dis / 4
                frame_len = 4
            else:
                frame_dis_dis = 1
                frame_len = int(frame_dis)
            frame_dis = int(frame_dis)
            frame_dis_dis = int(frame_dis_dis)
            for v_len in range(0, frame_len):
                if v_len == 0:
                    start = 0
                elif v_len == frame_len - 1:
                    start = frame_dis - 1
                else:
                    start = frame_dis_dis * v_len
                image_s = [os.path.realpath(os.path.join(imagedir, class_name, file_name, '{}.jpg'.format(image_index)))
                           for image_index in range(start, start + 16 * frame_dis, frame_dis)]
                train_data.append({'data': image_s, 'lable': label_count})

        for i in range(trainlen, len(file_list)):
            file_name = file_list[i]
            images_path = os.listdir(os.path.join(imagedir, class_name, file_name))
            images_len = len(images_path)
            frame_dis = images_len / 16
            if frame_dis >= 4:
                frame_dis_dis = frame_dis / 4
                frame_len = 4
            else:
                frame_dis_dis = 1
                frame_len = int(frame_dis)
            frame_dis = int(frame_dis)
            frame_dis_dis = int(frame_dis_dis)
            for v_len in range(0, frame_len):
                if v_len == 0:
                    start = 0
                elif v_len == frame_len - 1:
                    start = frame_dis - 1
                else:
                    start = frame_dis_dis * v_len
                image_s = [os.path.realpath(os.path.join(imagedir, class_name, file_name, '{}.jpg'.format(image_index)))
                           for image_index in range(start, start + 16 * frame_dis, frame_dis)]
                test_data.append({'data': image_s, 'lable': label_count})

        label_count += 1
    random.shuffle(train_data)
    random.shuffle(test_data)
    with open(labelpath, "wb+") as f:
        pickle.dump(label_class, f)
    with open(trainpath, "wb+") as f:
        pickle.dump(train_data, f)
    with open(testpath, "wb+") as f:
        pickle.dump(test_data, f)
    print (label_class)
    print ("train data length:", len(train_data))
    print ("test data length:", len(test_data))


if __name__ == '__main__':
    imagedir = args.imagedir
    labelpath = args.labelpath
    trainpath = args.trainpath
    testpath = args.testpath
    frac = args.frac
    createdatalist(imagedir, labelpath, trainpath, testpath, frac)
