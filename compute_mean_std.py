# coding: utf-8

import numpy as np
import cv2
import random
import os

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""
camera_means = []
camera_std = []
for camera_index in range(1,4):
    src_image_list = os.listdir('./camera_data_224')
        # 挑选多少图片进行计算
    src_image_list = sorted(src_image_list)

    img_h, img_w = 224, 224
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []

    for i in range(len(src_image_list)):
        if('camera'+str(camera_index) in src_image_list[i]):
            img = cv2.imread('./camera_data_224/'+src_image_list[i])
            img = cv2.resize(img, (img_h, img_w))

            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)
            if(i % 100 == 0):
                print('camera'+str(camera_index)+":",i)

    imgs = imgs.astype(np.float32)/255.


    for i in range(3):
        pixels = imgs[:,:,i,:].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse() # BGR --> RGB
    stdevs.reverse()

    camera_means.append(means)
    camera_std.append(stdevs)
    # print("normMean = {}".format(means))
    # print("normStd = {}".format(stdevs))
    # print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

print("camera_means:",camera_means)
print("camera_std:",camera_std)
