import numpy as np
import math

joint1_pos_init = np.array([0.000000,0.000000,0.200000])
joint2_pos_init = np.array([0.030000,0.000000,0.380000])
joint3_pos_init = np.array([0.030000,0.000000,0.720000])
joint4_pos_init = np.array([0.150000,0.000000,0.755000])



def angle1(joint2_pos):
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
    joint3_angle1_pos = joint2_pos + (joint3_pos_init - joint2_pos_init)
    _joint3_angle1_pos = np.array([joint3_angle1_pos[0], joint3_angle1_pos[1], 0.755000])
    _joint3_pos = joint3_pos * (0.720000 - 0.380000) / (0.755000 - 0.380000)

    joint4_angle1_pos = np.array([0.150000*math.cos(angle1), 0.150000*math.sin(angle1), 0.755000])

    joint4_angle2_pos = (np.array([joint4_angle1_pos[0] * math.cos(angle2), 
                                  joint4_angle1_pos[1] * math.cos(angle2), 
                                  joint4_angle1_pos[2] - (0.150000*math.sin(angle2))
                                  ])
                         - _joint3_angle1_pos) + _joint3_pos

    print(joint4_angle2_pos)

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


joint2_pos = np.array([0.029874,-0.002752,0.380000])
joint3_pos = np.array([-0.305885,0.028175,0.423698])
joint4_pos = np.array([-0.201347,0.018546,0.491550])
a1 = angle1(joint2_pos)
print(a1)
a2 = angle2(a1, joint2_pos, joint3_pos)
print(a2)
a3 = angle3(a1, a2, joint2_pos, joint3_pos, joint4_pos)
print(a3)

print("\n")

joint2_pos = np.array([-0.015225,0.025849,0.380000])
joint3_pos = np.array([-0.044914,0.076254,0.714930])
joint4_pos = np.array([-0.076971,0.130680,0.607064])
a1 = angle1(joint2_pos)
print(a1)
a2 = angle2(a1, joint2_pos, joint3_pos)
print(a2)
a3 = angle3(a1, a2, joint2_pos, joint3_pos, joint4_pos)
print(a3)


# 0 | 0.000000,0.000000,0.200000 | 0.030000,0.000000,0.380000 | 0.030000,0.000000,0.720000 | 0.150000,0.000000,0.755000 | 0.375000,0.000000,0.755000 | 0.465000,0.000000,0.755000
# 9 | 0.000000,0.000000,0.200000 | 0.029874,-0.002752,0.380000 | -0.305885,0.028175,0.423698 | -0.201347,0.018546,0.491550 | 0.013349,-0.001230,0.555890 | 0.033281,-0.088622,0.563964
# 10 | -0.000000,-0.000000,0.200000 | -0.015225,0.025849,0.380000 | -0.044914,0.076254,0.714930 | -0.076971,0.130680,0.607064 | -0.104776,0.177886,0.388836 | -0.146747,0.125017,0.448361

# 9 | 0.885814 | -0.091852 | -1.441917 | 2.063365 | -1.568093 | 1.242607
# 10 | 1.643832 | 2.103091 | 0.172913 | 2.810962 | 1.428433 | 3.119301