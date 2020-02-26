import numpy as np
import math
from scipy.linalg import solve
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import dsntnn

def angle1(joint2_pos):
    joint2_pos_init = np.array([0.030000,0.000000,0.380000])

    center = np.array([0.000000,0.000000,0.380000])
    rotate_axis = np.array([0.000000,0.000000,1.000000])

    vector1 = joint2_pos_init - center
    vector2 = joint2_pos - center

    direction = np.dot(rotate_axis, np.cross(vector1,vector2))
    # 0 is impossible
    if(direction < 0.0):
        direction = -1.0
    else:
        direction = 1.0

    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    angle = direction * math.acos(cos_angle)
    while(angle < -2.9670597284):
        angle += 2*math.pi
    while(angle > 2.9670597284):
        angle -= 2*math.pi

    return angle


def angle2(angle1, joint2_pos, joint3_pos):
    joint2_pos_init = np.array([0.030000,0.000000,0.380000])
    joint3_pos_init = np.array([0.030000,0.000000,0.720000])

    joint3_angle1_pos = joint2_pos + (joint3_pos_init - joint2_pos_init)

    center = joint2_pos
    rotate_axis = np.array([-math.sin(angle1), math.cos(angle1), 0.000000])

    vector1 = joint3_angle1_pos - center
    vector2 = joint3_pos - center

    direction = np.dot(rotate_axis, np.cross(vector1, vector2))
    # 0 is impossible
    if(direction < 0.0):
        direction = -1.0
    else:
        direction = 1.0

    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    angle = direction * math.acos(cos_angle)
    while(angle < -1.6580627894):
        angle += 2*math.pi
    while(angle > 2.3561944902):
        angle -= 2*math.pi

    return angle

def angle3(angle1, angle2, joint2_pos, joint3_pos, joint4_pos):
    # joint3_angle1_pos = np.array([joint2_pos[0], joint2_pos[1], 0.720000])
    _joint3_angle1_pos = np.array([joint2_pos[0], joint2_pos[1], 0.755000])
    _joint3_pos = joint2_pos + (joint3_pos-joint2_pos) / ((0.720000 - 0.380000) / (0.755000 - 0.380000))

    joint4_angle1_pos = np.array([0.150000*math.cos(angle1), 0.150000*math.sin(angle1), 0.755000])

    joint4_angle2_pos = (np.array([(joint4_angle1_pos[0] - 0.030000*math.cos(angle1)) * math.cos(angle2) + 0.030000*math.cos(angle1), 
                                  (joint4_angle1_pos[1] - 0.030000*math.sin(angle1)) * math.cos(angle2) + 0.030000*math.sin(angle1), 
                                  joint4_angle1_pos[2] - (0.120000*math.sin(angle2))
                                  ])
                         - _joint3_angle1_pos) + _joint3_pos

    center = joint3_pos
    rotate_axis = np.array([-math.sin(angle1), math.cos(angle1), 0.000000])

    vector1 = joint4_angle2_pos - center
    vector2 = joint4_pos - center

    direction = np.dot(rotate_axis, np.cross(vector1, vector2))
    # 0 is impossible
    if(direction < 0.0):
        direction = -1.0
    else:
        direction = 1.0

    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    angle = direction * math.acos(cos_angle)
    while(angle < -3.6651914292):
        angle += 2*math.pi
    while(angle > 1.1519173063):
        angle -= 2*math.pi

    return angle

def angle4_5(angle1, angle2, angle3, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos):
    angle4_list = []
    angle5_list = []
    # count angle5
    vector_5_4 = joint4_pos - joint5_pos
    vector_5_6 = joint6_pos - joint5_pos
    rotate_axis = np.cross(vector_5_6, vector_5_4)

    vector1 = (joint5_pos - joint4_pos) * (0.090000/0.225000)
    vector2 = joint6_pos - joint5_pos

    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle5 = math.acos(cos_angle)
    # while(angle5 < -2.3561944902):
    #     angle5 += 2*math.pi
    # while(angle5 > 2.3561944902):
    #     angle5 -= 2*math.pi

    # count angle4
    vector_5_4 = joint4_pos - joint5_pos
    vector_5_6 = joint6_pos - joint5_pos
    vector1 = np.array([-math.sin(angle1), math.cos(angle1), 0.000000])
    vector2 = np.cross(vector_5_6, vector_5_4)

    rotate_axis = joint5_pos - joint4_pos
    # if angle5 > 0
    direction = np.dot(rotate_axis, np.cross(vector1, vector2))
    # 0 is impossible
    if(direction < 0.0):
        direction = -1.0
    else:
        direction = 1.0
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle4 = direction * math.acos(cos_angle)
    # if(-2.9670597284 < angle4 < 2.9670597284):
    angle4_list.append(angle4)
    angle5_list.append(angle5)
    # if angle5 < 0
    vector1 = -vector1
    direction = np.dot(rotate_axis, np.cross(vector1, vector2))
    # 0 is impossible
    if(direction < 0.0):
        direction = -1.0
    else:
        direction = 1.0
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle4 = direction * math.acos(cos_angle)
    # if(-2.9670597284 < angle4 < 2.9670597284):
    angle4_list.append(angle4)
    angle5_list.append(-angle5)

    return angle4_list, angle5_list


def pixel2world(u_v_1, u_v_2, R_T_1, R_T_2, P_1, P_2):
    M_1 = np.dot(P_1, R_T_1)
    M_2 = np.dot(P_2, R_T_2)

    a = np.array([
        [u_v_1[0]*M_1[2][0]-M_1[0][0], u_v_1[0]*M_1[2][1]-M_1[0][1], u_v_1[0]*M_1[2][2]-M_1[0][2]],
        [u_v_1[1]*M_1[2][0]-M_1[1][0], u_v_1[1]*M_1[2][1]-M_1[1][1], u_v_1[1]*M_1[2][2]-M_1[1][2]],
        [u_v_2[0]*M_2[2][0]-M_2[0][0], u_v_2[0]*M_2[2][1]-M_2[0][1], u_v_2[0]*M_2[2][2]-M_2[0][2]],
        [u_v_2[1]*M_2[2][0]-M_2[1][0], u_v_2[1]*M_2[2][1]-M_2[1][1], u_v_2[1]*M_2[2][2]-M_2[1][2]]
    ])

    b = np.array([
        [M_1[0][3] - u_v_1[0]*M_1[2][3]],
        [M_1[1][3] - u_v_1[1]*M_1[2][3]],
        [M_2[0][3] - u_v_2[0]*M_2[2][3]],
        [M_2[1][3] - u_v_2[1]*M_2[2][3]],
    ])

    x = np.zeros((3,1))
    
    cv2.solve(a, b, x, flags=cv2.DECOMP_SVD)
    x = np.array(x).reshape((1,3))

    return x[0]

def model_predict2angle(u_v_1, u_v_2, u_v_3):
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

    joint_pos = np.zeros((5,3))
    for joint_index in range(5):
        joint_pos[joint_index] = (pixel2world(u_v_1[joint_index], u_v_2[joint_index], R_T[0], R_T[1], P[0], P[1]) + 
                                  pixel2world(u_v_1[joint_index], u_v_3[joint_index], R_T[0], R_T[2], P[0], P[2]) +
                                  pixel2world(u_v_2[joint_index], u_v_3[joint_index], R_T[1], R_T[2], P[1], P[2])) / 3
    print(joint_pos)

    joint2_pos = joint_pos[0]
    joint3_pos = joint_pos[1]
    joint4_pos = joint_pos[2]
    joint5_pos = joint_pos[3]
    joint6_pos = joint_pos[4]

    a1 = angle1(joint2_pos)
    a2 = angle2(a1, joint2_pos, joint3_pos)
    a3 = angle3(a1, a2, joint2_pos, joint3_pos, joint4_pos)
    a4, a5 = angle4_5(a1, a2, a3, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos)

    return a1, a2, a3, a4, a5




