# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     split_video
   Description :
   Author :       iffly
   date：          5/4/18
-------------------------------------------------
   Change Activity:
                   5/4/18:
-------------------------------------------------
"""
import argparse
import os
import re
import subprocess
import sys

import cv2
import numpy as np

parser = argparse.ArgumentParser(usage="python split_video.py --video videopath --output outputpath",
                                 description="help info.")
parser.add_argument("--video", default="", help="the video path.", dest="video_path", required=True)
parser.add_argument("--output", default="", help="the output path.", dest="output_path", required=True)
args = parser.parse_args()
video_path = args.video_path
output_path = args.output_path


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def split_video(video_path, class_name):
    _, file_name = os.path.split(video_path)
    video_name, _ = os.path.splitext(file_name)
    if not os.path.exists(os.path.join(output_path, 'frame', class_name, video_name)):
        os.makedirs(os.path.join(output_path, 'frame', class_name, video_name))
    if not os.path.exists(os.path.join(output_path, 'flow', class_name, video_name)):
        os.makedirs(os.path.join(output_path, 'flow', class_name, video_name))
    capture = cv2.VideoCapture(video_path)
    # format_str=os.popen('ffprobe -print_format  json -select_streams v -show_format -show_streams -i "{}"'.format(video_path))
    format_str = subprocess.Popen(
        'ffprobe -print_format  json -select_streams v -show_format -show_streams -i "{}"'.format(video_path),
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    format_str = format_str.stdout.read()
    # print format_str
    match = re.search('"rotate": "([0-9]+)"', format_str)
    rotate = 0
    if match:
        str1 = match.group(1)
        str2 = re.search('([0-9]+)', str1).group(1)
        rotate = int(str2)
    # ff = FFmpeg(inputs={video_path: video_path},
    #             outputs={out_path: 'fprobe -print_format  json -select_streams v -show_format -show_streams -i ,
    #                      out_path2: '-y -f mjpeg -ss 0 -t 0.001',
    #                      None: '-c copy -map 0 -y -f segment -segment_list {0} -segment_time 1  -bsf:v h264_mp4toannexb  {1}/cat_output%03d.ts'.format(
    #                          out_path3, base_path),
    #                      })
    if capture.isOpened():
        now_frame = 0
        old_frame = None
        while True:
            success, frame = capture.read()
            if not success:
                break
            # print frame.shape
            frame = rotate_bound(frame, rotate)
            cv2.imwrite(os.path.join(output_path, 'frame', class_name, video_name, "{}.jpg".format(now_frame)), frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calc optical
            # if  old_frame is None:
            #     old_frame=frame
            # optical_flow=cv2.calcOpticalFlowFarneback(old_frame, frame,None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # cv2.imwrite(os.path.join(output_path,'flow',class_name,video_name,"{}_h.png".format(now_frame)),optical_flow[:,:,0])
            # cv2.imwrite(os.path.join(output_path,'flow',class_name,video_name,"{}_v.png".format(now_frame)),optical_flow[:,:,1])
            now_frame += 1
        capture.release()
    else:
        print("cannot open " + video_name)


def process_data(video_path):
    class_list = os.listdir(video_path)
    print("class:", class_list)
    print("process start:")
    now_count = 0
    for class_name in class_list:
        if os.path.isdir(os.path.join(video_path, class_name)):
            video_list = os.listdir(os.path.join(video_path, class_name))
            for video_name in video_list:
                split_video(os.path.join(video_path, class_name, video_name), class_name)
                sys.stdout.write("{} process done.".format(now_count) + '\r')
                sys.stdout.flush()
                now_count += 1
    print("process end.")


if __name__ == '__main__':
    process_data(video_path)

    # print("123")
    # for i in range(0,10):
    #     sys.stdout.write("{} process done.".format(i)+'\r')
    #     sys.stdout.flush()
    #     time.sleep(2)
