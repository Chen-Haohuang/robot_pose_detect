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


img_h, img_w = 224, 224
batch_size = 2
lr_init = 0.001
max_epoch = 5

all_data_list = os.listdir('./camera_data')
test_data_list = random.sample(all_data_list, 1000)
train_data_list = list(set(all_data_list) - set(test_data_list))

joint_state_target = target_data_generate.gen_joint_state_target()
link_state_target = target_data_generate.gen_link_state_target()

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
							nn.Conv2d(512, 6, kernel_size=(1, 1), stride=(1, 1)),
							nn.Sigmoid()
							)
		)
		for i in range(self.N):
			self.stages.append(nn.Sequential(
								nn.Conv2d(262, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
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
								nn.Conv2d(512, 6, kernel_size=(1, 1), stride=(1, 1)),
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

class CustomLoss(torch.nn.Module):
	def __init__(self):
		super(CustomLoss,self).__init__()

	def forward(self,x,y):
		loss = torch.norm(x-y, p=1)
		return loss
		

class MyDataset(Dataset):
	def __init__(self, data_list, path, transform):
		imgs_index = data_list
		self.path = path
		self.imgs_index = imgs_index        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
		self.transform = transform

	def __getitem__(self, index):
		id = self.imgs_index[index]
		fn = self.path + id
		img = Image.open(fn).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
		img = self.transform(img)   # 在这里做transform，转为tensor等等

		matchObj = re.match(r'camera(.*?)-(.*?).jpg', id, re.M|re.I)
		camera_index = int(matchObj.group(1))
		data_index = int(matchObj.group(2))
		heatmap = target_data_generate.gen_one_heatmap_target(link_state_target, data_index, camera_index)

		return img, heatmap

	def __len__(self):
		return len(self.imgs_index)


means, stdevs = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
norm_trans = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=means, std=stdevs)
			])

train_data = MyDataset(train_data_list, './camera_data/', norm_trans)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = MyDataset(test_data_list, './camera_data/', norm_trans)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

net = RobotJointModel()
net.initialize_weights()

criterion = CustomLoss()                                                   # 选择损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略

for epoch in range(max_epoch):
	loss_sigma = 0.0    # 记录一个epoch的loss之和
	correct = 0.0
	total = 0.0
	scheduler.step()  # 更新学习率

	for i, data in enumerate(train_loader):
		inputs, labels = data
		inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)

		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
		if i % 10 == 9:
			loss_avg = loss_sigma / 10
			loss_sigma = 0.0
			print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
				epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg))

	# if epoch % 2 == 0:
	# 	loss_sigma = 0.0
	# 	net.eval()
	# 	for i, data in enumerate(test_loader):

	# 		# 获取图片和标签
	# 		images, labels = data
	# 		images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)

	# 		# forward
	# 		outputs = net(images)
	# 		outputs.detach_()

	# 		# 计算loss
	# 		loss = criterion(outputs, labels)
	# 		loss_sigma += loss.item()

	# 		# 统计
	# 		_, predicted = torch.max(outputs.data, 1)
	# 		# labels = labels.data    # Variable --> tensor


	# 	print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))