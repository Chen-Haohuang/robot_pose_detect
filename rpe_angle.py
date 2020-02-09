import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import target_data_generate
import re
import rpe_joint


class RobotAngleModel(nn.Module):

    def __init__(self):
        super(RobotAngleModel, self).__init__()
        self.stages = nn.Sequential(
                            nn.Linear(51, 512),
                            nn.ReLU(),
                            nn.Linear(512, 128),
                            nn.ReLU(),
                            nn.Linear(128, 6)
        )


    def forward(self, inputs):
        output = self.stages(inputs)
        
        return output

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0, 0.01)
            m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.stages:
            self._initialize_weights(m)

class MyDataset(Dataset):
    def __init__(self, data_list):
        imgs_index = data_list
        self.imgs_index = imgs_index        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.link_state_target = target_data_generate.gen_link_state_target()
        self.joint_state_target = target_data_generate.gen_joint_state_target()

    def __getitem__(self, index):
        data_index = self.imgs_index[index]

        data = []

        for camera_index in range(1,4):
            if( camera_index == 1):
                u_v_0 = [28.125,28.125]
                data += [28.125,28.125]
            else:
                u_v_0 = [28.125,34.79884317]
                data += [28.125,34.79884317]
            u_v,_ = target_data_generate.gen_one_heatmap_target(self.link_state_target, data_index, camera_index)
            for j in range(5):
                if(j == 0):
                    tan_value = (u_v[j][1]-u_v_0[1]) / (u_v[j][0]-u_v_0[0])
                else:
                    tan_value = (u_v[j][1]-u_v[j-1][1]) / (u_v[j][0]-u_v[j-1][0])
                data.append(tan_value)
                data.append(u_v[j][0])
                data.append(u_v[j][1])

        data = np.array(data, dtype=np.float32)
        label = self.joint_state_target[data_index]
        label = np.array(label, dtype=np.float32)

        return data , label, data_index

    def __len__(self):
        return len(self.imgs_index)
