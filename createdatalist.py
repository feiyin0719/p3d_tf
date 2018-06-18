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
            image_s = [[], [], [], []]

            for image_index in range(len(images_path)):
                temppath = os.path.join(imagedir, class_name, file_name, '{}.jpg'.format(image_index))
                image_s[0].append(temppath)
                if image_index % 2 == 0:
                    image_s[1].append(temppath)
                if image_index % 4 == 0:
                    image_s[2].append(temppath)
                if image_index % 8 == 0:
                    image_s[3].append(temppath)
                for j in range(0, 4):
                    if len(image_s[j]) == 16:
                        train_data.append({'data': image_s[j], 'label': label_count})
                        # train_data['data'].append(image_s[j])
                        # train_data['label'].append(label_count)
                        image_s[j] = []
        for i in range(trainlen, len(file_list)):
            file_name = file_list[i]
            images_path = os.listdir(os.path.join(imagedir, class_name, file_name))
            image_s = [[], [], [], []]

            for image_index in range(len(images_path)):
                temppath = os.path.realpath(os.path.join(imagedir, class_name, file_name, '{}.jpg'.format(image_index)))
                image_s[0].append(temppath)
                if image_index % 2 == 0:
                    image_s[1].append(temppath)
                if image_index % 4 == 0:
                    image_s[2].append(temppath)
                if image_index % 8 == 0:
                    image_s[3].append(temppath)
                for j in range(0, 4):
                    if len(image_s[j]) == 16:
                        test_data.append({'data': image_s[j], 'label': label_count})
                        # test_data['data'].append(image_s[j])
                        # test_data['label'].append(label_count)
                        image_s[j] = []

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
