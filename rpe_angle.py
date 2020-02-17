import numpy as np
import math
import target_data_generate
from scipy.linalg import solve
import cv2
import rpe_model
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import dsntnn

torch.set_printoptions(threshold=100000000,linewidth=100000000000)
np.set_printoptions(threshold=100000000,linewidth=100000000000)

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

    return angle

def angle4(angle1, angle2, angle3, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos):

    rotate_axis = np.array([-math.cos(angle1), -math.sin(angle1), 0.000000])

    vector_5_4 = joint4_pos - joint5_pos
    vector_5_6 = joint6_pos - joint5_pos
    vector1 = np.array([-math.sin(angle1), math.cos(angle1), 0.000000])
    vector2 = np.cross(vector_5_6, vector_5_4)

    direction = np.dot(rotate_axis, np.cross(vector1, vector2))
    # 0 is impossible
    if(direction < 0.0):
        direction = -1.0
    else:
        direction = 1.0

    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    angle = direction * math.acos(cos_angle)

    return angle


def angle5(angle1, angle2, angle3, angle4, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos):

    vector_5_4 = joint4_pos - joint5_pos
    vector_5_6 = joint6_pos - joint5_pos
    rotate_axis = np.cross(vector_5_6, vector_5_4)

    vector1 = (joint5_pos - joint4_pos) * (0.090000/0.225000)
    vector2 = joint6_pos - joint5_pos

    direction = np.dot(rotate_axis, np.cross(vector1, vector2))
    # 0 is impossible
    if(direction < 0.0):
        direction = -1.0
    else:
        direction = 1.0

    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    angle = direction * math.acos(cos_angle)

    return angle


def pixel2world(u_v_1, u_v_2, R_T_1, R_T_2, P_1, P_2):
    M_1 = np.dot(P_1, R_T_1)
    M_2 = np.dot(P_2, R_T_2)

    a = np.array([
        [u_v_1[0]*M_1[2][0]-M_1[0][0], u_v_1[0]*M_1[2][1]-M_1[0][1], u_v_1[0]*M_1[2][2]-M_1[0][2]],
        [u_v_1[1]*M_1[2][0]-M_1[1][0], u_v_1[1]*M_1[2][1]-M_1[1][1], u_v_1[1]*M_1[2][2]-M_1[1][2]],
        [u_v_2[0]*M_2[2][0]-M_2[0][0], u_v_2[0]*M_2[2][1]-M_2[0][1], u_v_2[0]*M_2[2][2]-M_2[0][2]]#,
        #[u_v_2[1]*M_2[2][0]-M_2[1][0], u_v_2[1]*M_2[2][1]-M_2[1][1], u_v_2[1]*M_1[2][2]-M_2[1][2]]
    ])

    b = np.array([
        [M_1[0][3] - u_v_1[0]*M_1[2][3]],
        [M_1[1][3] - u_v_1[1]*M_1[2][3]],
        [M_2[0][3] - u_v_2[0]*M_2[2][3]]#,
        #[M_2[1][3] - u_v_2[1]*M_2[2][3]],
    ])

    x = solve(a, b)
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
    a4 = angle4(a1, a2, a3, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos)
    a5 = angle5(a1, a2, a3, a4, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos)

    print(a1, a2, a3, a4, a5)

    for joint_index in range(5):
        joint_pos[joint_index] = pixel2world(u_v_1[joint_index], u_v_2[joint_index], R_T[0], R_T[1], P[0], P[1])
    print(joint_pos)

    joint2_pos = joint_pos[0]
    joint3_pos = joint_pos[1]
    joint4_pos = joint_pos[2]
    joint5_pos = joint_pos[3]
    joint6_pos = joint_pos[4]

    a1 = angle1(joint2_pos)
    a2 = angle2(a1, joint2_pos, joint3_pos)
    a3 = angle3(a1, a2, joint2_pos, joint3_pos, joint4_pos)
    a4 = angle4(a1, a2, a3, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos)
    a5 = angle5(a1, a2, a3, a4, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos)

    print(a1, a2, a3, a4, a5)

    return a1, a2, a3, a4, a5


data_index = '10'
means, stdevs = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
norm_trans = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=means, std=stdevs)
			])
img_1 = Image.open('./camera_train_data/camera1-'+data_index+'.png').convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
img_1 = norm_trans(img_1)   # 在这里做transform，转为tensor等等
img_2 = Image.open('./camera_train_data/camera2-'+data_index+'.png').convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
img_2 = norm_trans(img_2)   # 在这里做transform，转为tensor等等
img_3 = Image.open('./camera_train_data/camera3-'+data_index+'.png').convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
img_3 = norm_trans(img_3)   # 在这里做transform，转为tensor等等

net = rpe_model.RobotJointModel()
net.load_state_dict(torch.load('joint_model_net_DSNT_1e-5_50.pth'))
net.eval()
img_1 = img_1.unsqueeze(0)
heatmap_1 = net(img_1)
c_1 = dsntnn.dsnt(heatmap_1)
img_2 = img_2.unsqueeze(0)
heatmap_2 = net(img_2)
c_2 = dsntnn.dsnt(heatmap_2)
img_3 = img_3.unsqueeze(0)
heatmap_3 = net(img_3)
c_3 = dsntnn.dsnt(heatmap_3)

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(sum(heatmap_1.detach().numpy()[0][:]))
# plt.subplot(1,3,2)
# plt.imshow(sum(heatmap_2.detach().numpy()[0][:]))
# plt.subplot(1,3,3)
# plt.imshow(sum(heatmap_3.detach().numpy()[0][:]))
# plt.show()


link_state_target = target_data_generate.gen_link_state_target()
u_v_1_data = target_data_generate.gen_one_heatmap_target(link_state_target, int(data_index), 1)
u_v_2_data = target_data_generate.gen_one_heatmap_target(link_state_target, int(data_index), 2)
u_v_3_data = target_data_generate.gen_one_heatmap_target(link_state_target, int(data_index), 3)

# print(((c_1+1)*224-1)/2)
# print(u_v_1_data)
# print(((c_2+1)*224-1)/2)
# print(u_v_2_data)
# print(((c_3+1)*224-1)/2)
# print(u_v_3_data)

c_1 = ((c_1+1)*224-1)/2
c_2 = ((c_2+1)*224-1)/2
c_3 = ((c_3+1)*224-1)/2
print(model_predict2angle(c_1.detach().numpy()[0], c_2.detach().numpy()[0], c_3.detach().numpy()[0]))



# 0 | 0.000000,0.000000,0.200000 | 0.030000,0.000000,0.380000 | 0.030000,0.000000,0.720000 | 0.150000,0.000000,0.755000 | 0.375000,0.000000,0.755000 | 0.465000,0.000000,0.755000
# 9 | 0.000001,-0.000001,0.200000 | 0.024929,0.016691,0.380000 | -0.128533,-0.086090,0.665457 | -0.200413,-0.134228,0.575230 | -0.362407,-0.242716,0.462920 | -0.337064,-0.182118,0.401393
# 10 | -0.000000,-0.000000,0.200000 | 0.013885,-0.026593,0.380000 | -0.131905,0.252607,0.508033 | -0.150545,0.288304,0.626368 | -0.155151,0.297121,0.851148 | -0.140972,0.367226,0.905776


# 9 | 0.590127 | -0.574278 | -3.089930 | 2.719269 | 1.758030 | 3.230354
# 10 | -1.089569 | -1.184705 | -0.430301 | 2.438517 | 0.884482 | -0.452043


# joint2_pos = np.array([0.024929,0.016691,0.380000])
# joint3_pos = np.array([-0.128533,-0.086090,0.665457])
# joint4_pos = np.array([-0.200413,-0.134228,0.575230])
# joint5_pos = np.array([-0.362407,-0.242716,0.462920])
# joint6_pos = np.array([-0.337064,-0.182118,0.401393])
# a1 = angle1(joint2_pos)
# print(a1)
# a2 = angle2(a1, joint2_pos, joint3_pos)
# print(a2)
# a3 = angle3(a1, a2, joint2_pos, joint3_pos, joint4_pos)
# print(a3)
# a4 = angle4(a1, a2, a3, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos)
# print(a4)
# a5 = angle5(a1, a2, a3, a4, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos)
# print(a5)

# print("\n")

# joint2_pos = np.array([0.013885,-0.026593,0.380000])
# joint3_pos = np.array([-0.131905,0.252607,0.508033])
# joint4_pos = np.array([-0.150545,0.288304,0.626368])
# joint5_pos = np.array([-0.155151,0.297121,0.851148])
# joint6_pos = np.array([-0.140972,0.367226,0.905776])
# a1 = angle1(joint2_pos)
# print(a1)
# a2 = angle2(a1, joint2_pos, joint3_pos)
# print(a2)
# a3 = angle3(a1, a2, joint2_pos, joint3_pos, joint4_pos)
# print(a3)
# a4 = angle4(a1, a2, a3, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos)
# print(a4)
# a5 = angle5(a1, a2, a3, a4, joint2_pos, joint3_pos, joint4_pos, joint5_pos, joint6_pos)
# print(a5)