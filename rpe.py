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


img_h, img_w = 224, 224
batch_size = 2

class RobotJointModel(nn.Module):

	def __init__(self):
		super(RobotJointModel, self).__init__()
		self.N = 2
		self.M = 1
		self.camera_num = 3
		self.features = nn.ModuleList()
		for i in range(self.camera_num):
			self.features.append(nn.Sequential(
									torchvision.models.vgg16(pretrained=True).features[:12]
								)
			)
		self.stages = nn.ModuleList()
		for i in range(self.camera_num):
			self.stages.append(nn.ModuleList())
			self.stages[i].append(nn.Sequential(
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
									nn.Conv2d(512, 7, kernel_size=(1, 1), stride=(1, 1)),
									nn.Sigmoid()
								  )
			)
			for j in range(self.N):
				self.stages[i].append(nn.Sequential(
										nn.Conv2d(263, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
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
										nn.Conv2d(512, 7, kernel_size=(1, 1), stride=(1, 1)),
										nn.Sigmoid()
									 )
   				)

			self.connection = nn.Sequential(
									nn.Conv2d(789, 512, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=False),
									nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
									nn.ReLU(),
									nn.Conv2d(512, 128, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=False),
									nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
							   )

			self.jointDetect = nn.Sequential(
									nn.Linear(86528, 1000),
									nn.ReLU(),
									nn.Linear(1000, 100),
									nn.ReLU(),
									nn.Linear(100, 500),
									nn.ReLU(),
									nn.Linear(500, 6),
									nn.ReLU()
								)




	def forward(self, inputs):
		stage_output_list = []
		for i in range(self.camera_num):
			VGG_output = self.features[i](inputs[i])
			for j in range(self.N+1):
				if(j == 0):
					stage_output = self.stages[i][j](VGG_output)
					stage_input = torch.cat((VGG_output, stage_output), 1)
				else:
					stage_output = self.stages[i][j](stage_input)
					stage_input = torch.cat((VGG_output, stage_output), 1)

			stage_output_list.append(stage_input)
		
		for i in range(self.camera_num):
			if(i == 0):
				stage_output = stage_output_list[i]
			else:
				stage_output = torch.cat((stage_output, stage_output_list[1]), 1)

		stage_output = self.connection(stage_output)
		
		detect_input = stage_output.view(batch_size, 86528)

		output = self.jointDetect(detect_input)

		return output


class MyDataset(Dataset):
	def __init__(self, path, transform):
		imgs_index = random.sample([i for i in range(5000)], 10)
		self.path = path
		self.imgs_index = imgs_index        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
		self.transform = transform

	def __getitem__(self, index):
		id = self.imgs_index[index]
		img = []
		for camera_index in range(1,4):
			fn = self.path+'camera'+str(camera_index)+'-'+str(id)+'.jpg'
			img_i = Image.open(fn).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
			img_i = self.transform(img_i)   # 在这里做transform，转为tensor等等
			img.append(img_i)
		return img

	def __len__(self):
		return len(self.imgs_index)


means, stdevs = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
norm_trans = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=means, std=stdevs)
			])
		
train_data = MyDataset('./camera_data_224/', norm_trans)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

net = RobotJointModel()
print(net)
for data in train_loader:
	out = net(data)
	print(out)
