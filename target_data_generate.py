import re
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

joint_state_file = open("joints_controller_cmd_publisher-3-stdout.log","r")
line = joint_state_file.readline()
joint_state_map = {}
while(line):
    line = line.strip().strip('\x1b[0m')
    matchObj = re.match(r'.*?:(.*)',line, re.M|re.I)
    data = matchObj.group(1).strip().split(' | ')
    joint_state_map[int(data[0])] = [float(d) for d in data[1:]]
    line = joint_state_file.readline()
joint_state_file.close()


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

link_state_file = open("link_state_saver-4-stdout.log","r")
line = link_state_file.readline()
link_state_map = {}
while(line):
    line = line.strip().strip('\x1b[0m')
    matchObj = re.match(r'.*?:(.*)',line, re.M|re.I)
    data = matchObj.group(1).strip().split(' | ')
    pose = []
    for d_str in data[1:]:
        d = d_str.split(',')
        pose.append([float(_d) for _d in d])
    link_state_map[int(data[0])] = pose
    line = link_state_file.readline()
link_state_file.close()

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


for i in range(5, len(link_state_map)+1):
    link_data = link_state_map[i]
    print(link_data)
    for k in range(3):
        heatmap = np.zeros((224,224))
        for j in range(len(link_data)):
            W = np.array([link_data[j] + [1.0]])
            W = np.transpose(W)
            C = np.dot(R_T[k], W)
            Zc = C[2][0]
            u_v = np.dot(P[k], C)
            u_v = u_v / Zc
            u_v = u_v / u_v[2][0]
            
            print(u_v)
            # ax = plt.gca()
            # # ax.xaxis.set_ticks_position('top')
            # ax.spines["bottom"].set_position
            heatmap += CenterGaussianHeatMap(224, 224, u_v[0][0], u_v[1][0], 3)
            
        cv2.imshow("hm",heatmap)
        cv2.waitKey(0)

    break

