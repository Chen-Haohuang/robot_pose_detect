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

class RobotJointModel(nn.Module):

	def __init__(self):
		super(RobotJointModel, self).__init__()
		self.N = 5
		self.M = 1
		self.features = torchvision.models.vgg16(pretrained=True).features[:12]
		self.stages = nn.ModuleList()
		self.stages.append(nn.Sequential(
							nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
							nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
							nn.ReLU(),
							nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
							nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
							nn.ReLU(),
							nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
							nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
							nn.ReLU(),
							nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1)),
							nn.ReLU(),
							nn.Conv2d(512, 5, kernel_size=(1, 1), stride=(1, 1)),
							nn.Sigmoid()
							)
		)
		for i in range(self.N):
			self.stages.append(nn.Sequential(
								nn.Conv2d(261, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
								nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
								nn.ReLU(),
								nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
								nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
								nn.ReLU(),
								nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
								nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
								nn.ReLU(),
								nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
								nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
								nn.ReLU(),
								nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
								nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
								nn.ReLU(),
								nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1)),
								nn.ReLU(),
								nn.Conv2d(512, 5, kernel_size=(1, 1), stride=(1, 1)),
								nn.Sigmoid()
								)
			)


	def forward(self, inputs):
		VGG_output = self.features(inputs)
		for i in range(self.N+1):
			if(i == 0):
				stage_output = self.stages[i](VGG_output)
				stage_input = torch.cat((VGG_output, stage_output), 1)
			else:
				stage_output = self.stages[i](stage_input)
				stage_input = torch.cat((VGG_output, stage_output), 1)
		
		return stage_output

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
		for s in self.stages:
			for m in s:
				self._initialize_weights(m)
		

class MyDataset(Dataset):
    def __init__(self, data_list, path, transform):
        imgs_index = data_list
        self.path = path
        self.imgs_index = imgs_index        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.link_state_target = target_data_generate.gen_link_state_target()

    def __getitem__(self, index):
        id = self.imgs_index[index]
        fn = self.path + id
        img = Image.open(fn).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img = self.transform(img)   # 在这里做transform，转为tensor等等

        matchObj = re.match(r'camera(.*?)-(.*?).png', id, re.M|re.I)
        camera_index = int(matchObj.group(1))
        data_index = int(matchObj.group(2))
        _, heatmap = target_data_generate.gen_one_heatmap_target(self.link_state_target, data_index, camera_index)

        return img, heatmap, id

    def __len__(self):
        return len(self.imgs_index)

