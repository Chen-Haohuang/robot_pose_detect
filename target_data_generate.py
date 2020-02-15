import re
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json

def gen_joint_state_target():
    joint_state_file = open("joints_controller_cmd_publisher-stdout.log","r")
    line = joint_state_file.readline()
    joint_state = []
    while(line):
        line = line.strip().strip('\x1b[0m')
        matchObj = re.match(r'.*?:(.*)',line, re.M|re.I)
        data = matchObj.group(1).strip().split(' | ')
        one_state = np.array([[float(d) for d in data[1:]]])
        one_state = np.transpose(one_state)
        joint_state.append(one_state)
        line = joint_state_file.readline()
    joint_state_file.close()
    return joint_state

def gen_link_state_target():
    link_state_file = open("link_state_saver-stdout.log","r")
    line = link_state_file.readline()
    link_state = []
    while(line):
        line = line.strip().strip('\x1b[0m')
        matchObj = re.match(r'.*?:(.*)',line, re.M|re.I)
        data = matchObj.group(1).strip().split(' | ')
        pose = []
        for d_str in data[1:]:
            d = d_str.split(',')
            pose.append([float(_d) for _d in d])
        link_state.append(pose)
        line = link_state_file.readline()
    link_state_file.close()
    return link_state

def gen_one_heatmap_target(link_state, data_index, camera_index):

    def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
        X1 = np.linspace(1, img_width, img_width)
        Y1 = np.linspace(1, img_height, img_height)
        [X, Y] = np.meshgrid(X1, Y1)
        X = X - c_x
        Y = Y - c_y
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma * sigma
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        return heatmap
    # Compute gaussian kernel
    def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
        gaussian_map = np.zeros((img_height, img_width))
        for x_p in range(img_width):
            for y_p in range(img_height):
                dist_sq = (x_p - c_x) * (x_p - c_x) + (y_p - c_y) * (y_p - c_y)
                exponent = dist_sq / 2.0 / variance / variance
                gaussian_map[y_p, x_p] = np.exp(-exponent)
        return gaussian_map

    # R1_x = np.array([
    #             [1.0, 0.0, 0.0],
    #             [0.0, 0.0, 1.0],
    #             [0.0, -1.0, 0.0]
    #             ])
    # R1_y = np.array([
    #             [1.0, 0.0, 0.0],
    #             [0.0, 1.0, 0.0],
    #             [0.0, 0.0, 1.0]
    #             ])
    # R1_z = np.array([
    #             [0.0, -1.0, 0.0],
    #             [1.0, 0.0, 0.0],
    #             [0.0, 0.0, 1.0]
    #             ])
    # print(np.dot(np.dot(R1_x,R1_y),R1_z))
    # R2_x = np.array([
    #             [1.0, 0.0, 0.0],
    #             [0.0, 0.0, 1.0],
    #             [0.0, -1.0, 0.0]
    #             ])
    # R2_y = np.array([
    #             [1.0, 0.0, 0.0],
    #             [0.0, 1.0, 0.0],
    #             [0.0, 0.0, 1.0]
    #             ])
    # R2_z = np.array([
    #             [-1.0, 0.0, 0.0],
    #             [0.0, -1.0, 0.0],
    #             [0.0, 0.0, 1.0]
    #             ])
    # print(np.dot(np.dot(R2_x,R2_y),R2_z))
    # R3_x = np.array([
    #             [1.0, 0.0, 0.0],
    #             [0.0, -1.0, 0.0],
    #             [0.0, 0.0, -1.0]
    #             ])
    # R3_y = np.array([
    #             [1.0, 0.0, 0.0],
    #             [0.0, 1.0, 0.0],
    #             [0.0, 0.0, 1.0]
    #             ])
    # R3_z = np.array([
    #             [0.0, -1.0, 0.0],
    #             [1.0, 0.0, 0.0],
    #             [0.0, 0.0, 1.0]
    #             ])
    # print(np.dot(np.dot(R3_x,R3_y),R3_z))

    R_T = [
            np.array([
                [0.0, -1.0, 0.0, 0.0], 
                [0.0, 0.0, 1.0, -0.5], 
                [-1.0, 0.0, 0.0, -1.5],
                [0.0, 0.0, 0.0, 1.0]
                ]),
            np.array([
                [-1.0, 0.0, 0.0, 0.0], 
                [0.0, 0.0, 1.0, -0.5], 
                [0.0, 1.0, 0.0, -1.5],
                [0.0, 0.0, 0.0, 1.0]
                ]),
            np.array([
                [0.0, -1.0, 0.0, 0.0], 
                [-1.0, 0.0, 0.0, 0.0], 
                [0.0, 0.0, -1.0, 1.85],
                [0.0, 0.0, 0.0, 1.0]
                ])
        ]

    P = [
            np.array([
                [133.47686340839743, 0.0, 112.5, 0.0], 
                [0.0, 133.47686340839743, 112.5, 0.0],
                [0.0, 0.0, 1.0, 0.0]
                ]),
            np.array([
                [133.47686340839743, 0.0, 112.5, 0.0], 
                [0.0, 133.47686340839743, 112.5, 0.0],
                [0.0, 0.0, 1.0, 0.0]
                ]),
            np.array([
                [133.47686340839743, 0.0, 112.5, 0.0], 
                [0.0, 133.47686340839743, 112.5, 0.0],
                [0.0, 0.0, 1.0, 0.0]
                ])
        ]

    link_data = link_state[data_index]
    heatmap = np.zeros((5,56,56))
    imghm = np.zeros((56,56))
    u_v_ret = np.zeros((5,2))
    for j in range(len(link_data)):
        if( j == 0 ):
            continue
        W = np.array([link_data[j] + [1.0]])
        W = np.transpose(W)
        C = np.dot(R_T[camera_index-1], W)
        Zc = C[2][0]
        u_v = np.dot(P[camera_index-1], C)
        u_v = u_v / Zc
        u_v = u_v / u_v[2][0]
        heatmap[j-1] = cv2.resize(CenterGaussianHeatMap(224, 224, u_v[0][0], u_v[1][0], 3), (56,56))
        imghm += heatmap[j-1]
        u_v_ret[j-1][0] = u_v[1][0]
        u_v_ret[j-1][1] = u_v[0][0]

    return u_v_ret, heatmap
        

