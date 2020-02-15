import numpy as np
import math
import target_data_generate
from scipy.linalg import solve
import cv2

np.set_printoptions(threshold=100000000)

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

X = np.zeros((56,56))
Y = np.zeros((56,56))
for i in range(56):
    for j in range(56):
        X[i][j] = (2.0*j-57.0)/56.0
        Y[i][j] = (2.0*i-57.0)/56.0

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
        u_v = np.zeros((5,2))
        u_v[0] = u_v_1[joint_index]*4.0
        u_v[1] = u_v_2[joint_index]*4.0
        u_v[2] = u_v_3[joint_index]*4.0
        print("=====",u_v[0],u_v[1],u_v[2])
        print(pixel2world(u_v[0], u_v[1], R_T[0], R_T[1], P[0], P[1]))
        print(pixel2world(u_v[0], u_v[2], R_T[0], R_T[2], P[0], P[2]))
        print(pixel2world(u_v[1], u_v[2], R_T[1], R_T[2], P[1], P[2]))
        joint_pos[joint_index] = (pixel2world(u_v[0], u_v[1], R_T[0], R_T[1], P[0], P[1]) + 
                                  pixel2world(u_v[0], u_v[2], R_T[0], R_T[2], P[0], P[2]) +
                                  pixel2world(u_v[1], u_v[2], R_T[1], R_T[2], P[1], P[2])) / 3.0
        print(joint_pos[joint_index])

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

    return a1, a2, a3, a4, a5


link_state_target = target_data_generate.gen_link_state_target()
u_v_1, heatmap_1 = target_data_generate.gen_one_heatmap_target(link_state_target, 9, 1)
u_v_2, heatmap_2 = target_data_generate.gen_one_heatmap_target(link_state_target, 9, 2)
u_v_3, heatmap_3 = target_data_generate.gen_one_heatmap_target(link_state_target, 9, 3)

print(u_v_1[0]*4.0, u_v_2[0]*4.0, u_v_3[0]*4.0)
print("!!!!!!!!!!!",pixel2world(u_v_1[0]*4.0, u_v_3[0]*4.0, R_T[0], R_T[2], P[0], P[2]))

# print(u_v_1)
# print(u_v_2)
# print(u_v_3)

a = model_predict2angle(u_v_1, u_v_2, u_v_3)
print(a)

link_state_target = target_data_generate.gen_link_state_target()
u_v_1, _ = target_data_generate.gen_one_heatmap_target(link_state_target, 9, 1)
u_v_2, _ = target_data_generate.gen_one_heatmap_target(link_state_target, 9, 2)

pixel2world(u_v_1[0]*4.0, u_v_2[0]*4.0, R_T[0], R_T[1], P[0], P[1])



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