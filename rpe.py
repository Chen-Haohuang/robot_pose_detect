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
import dsntnn
# torch.set_printoptions(threshold=100000000000, linewidth=10000000000)

use_gpu = torch.cuda.is_available()
torch.set_num_threads(80)

img_h, img_w = 224, 224
batch_size = 16
lr_init = 1e-5
num_workers_init = 64
max_epoch = 50

all_data_list = os.listdir('./camera_train_data')
test_data_list = random.sample(all_data_list, 2000)
train_data_list = list(set(all_data_list) - set(test_data_list))
random.shuffle(test_data_list)
random.shuffle(train_data_list)

joint_state_target = target_data_generate.gen_joint_state_target()
link_state_target = target_data_generate.gen_link_state_target()

class RobotJointModel(nn.Module):

	def __init__(self):
		super(RobotJointModel, self).__init__()
		self.N = 5
		self.M = 1
		self.joint_num = 5
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
				
		heatmaps = dsntnn.flat_softmax(stage_output)
		coords = dsntnn.dsnt(heatmaps)

		return coords, heatmaps

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

	def __getitem__(self, index):
		id = self.imgs_index[index]
		fn = self.path + id

		img = torch.from_numpy(cv2.imread(fn)).permute(2,0,1).float()
		img = img.div(255)

		matchObj = re.match(r'camera(.*?)-(.*?).png', id, re.M|re.I)
		camera_index = int(matchObj.group(1))
		data_index = int(matchObj.group(2))
		coords, heatmaps = target_data_generate.gen_one_heatmap_target(link_state_target, data_index, camera_index)

		return img, coords, heatmaps, id

	def __len__(self):
		return len(self.imgs_index)


means, stdevs = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
norm_trans = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=means, std=stdevs)
			])

train_data = MyDataset(train_data_list, './camera_train_data/', norm_trans)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers_init)

test_data = MyDataset(test_data_list, './camera_train_data/', norm_trans)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers_init)

net = RobotJointModel()
if(use_gpu):
	net = net.cuda()
net.initialize_weights()

criterion = nn.SmoothL1Loss()                                                   # 选择损失函数
if(use_gpu):
	criterion = criterion.cuda()
#optimizer = torch.optim.SGD(net.parameters(), lr=1e-5, momentum=0.9, dampening=0.1)    # 选择优化器
optimizer = torch.optim.Adam(net.parameters(), lr=lr_init)    # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)     # 设置学习率下降策略
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, threshold=1e-7, min_lr=1e-7, factor=0.9)     # 设置学习率下降策略

for epoch in range(max_epoch):
	euc_loss_sigma = 0.0
	reg_loss_sigma = 0.0    # 记录一个epoch的loss之和
	loss_sigma = 0.0
	correct = 0.0
	total = 0.0
	pre_i = -1

	for i, data in enumerate(train_loader):
		inputs, coord_labels, heatmaps_labels, _ = data
		
		inputs, coord_labels, heatmaps_labels = torch.autograd.Variable(inputs), torch.autograd.Variable(coord_labels), torch.autograd.Variable(heatmaps_labels)

		coord_labels = (coord_labels*2 + 1) / torch.Tensor([img_w,img_h]) - 1

		if(use_gpu):
			inputs, coord_labels, heatmaps_labels = inputs.cuda(), coord_labels.cuda(), heatmaps_labels.cuda()

		coords, heatmaps = net(inputs)

		euc_losses = dsntnn.euclidean_losses(coords, coord_labels)
		reg_losses = dsntnn.js_reg_losses(heatmaps, coord_labels, sigma_t=1.0)
		loss = dsntnn.average_loss(euc_losses + reg_losses)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if(use_gpu):
			euc_loss = euc_losses.cpu()
			reg_loss = reg_losses.cpu()
			loss = loss.cpu()
		euc_loss_sigma += dsntnn.average_loss(euc_losses).item()
		reg_loss_sigma += dsntnn.average_loss(reg_losses).item()
		loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
		if i % 10 == 9 or i == len(train_loader)-1:
			euc_loss_avg = euc_loss_sigma / (i-pre_i)
			reg_loss_avg = reg_loss_sigma / (i-pre_i)
			loss_avg = loss_sigma / (i-pre_i)
			pre_i = i
			euc_loss_sigma = 0.0
			reg_loss_sigma = 0.0
			loss_sigma = 0.0
			print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] euc_losses: {:.8f} reg_losses: {:.8f} loss: {:.8f}".format(
				epoch + 1, max_epoch, i + 1, len(train_loader), euc_loss_avg, reg_loss_avg, loss_avg))
		#scheduler.step()

	euc_loss_sigma = 0.0
	reg_loss_sigma = 0.0
	loss_sigma = 0.0
	net.eval()
	for i, data in enumerate(test_loader):
		images, coord_labels, heatmaps_labels, name = data
		images, coord_labels, heatmaps_labels = torch.autograd.Variable(images), torch.autograd.Variable(coord_labels), torch.autograd.Variable(heatmaps_labels)

		coord_labels = (coord_labels*2 + 1) / torch.Tensor([img_w,img_h]) - 1

		if(use_gpu):
			images, coord_labels, heatmaps_labels = images.cuda(), coord_labels.cuda(), heatmaps_labels.cuda()

		# forward
		coords, heatmaps = net(images)
		
		out_for_image = heatmaps*255
		out_for_image = out_for_image.cpu()
		for b in range(len(out_for_image)):
			joint_image = np.zeros((56,56))
			for j in range(5):
				joint_image += out_for_image.detach().numpy()[b][j]
			cv2.imwrite('./test_predict/'+name[b]+'-joints.png', joint_image)

		# 计算loss
		euc_losses = dsntnn.euclidean_losses(coords, coord_labels)
		reg_losses = dsntnn.js_reg_losses(heatmaps, coord_labels, sigma_t=1.0)
		loss = dsntnn.average_loss(euc_losses + reg_losses)

		euc_loss_sigma += dsntnn.average_loss(euc_losses).item()
		reg_loss_sigma += dsntnn.average_loss(reg_losses).item()
		loss_sigma += loss.item()

	euc_loss_avg = euc_loss_sigma / len(test_loader)
	reg_loss_avg = reg_loss_sigma / len(test_loader)
	loss_avg = loss_sigma / len(test_loader)
	print("Testing: Epoch[{:0>3}/{:0>3}] euc_losses: {:.8f} reg_losses: {:.8f}  loss: {:.8f}".format(
		epoch + 1, max_epoch, euc_loss_avg, reg_loss_avg, loss_avg))

PATH = 'joint_model_net.pth'
torch.save(net.state_dict(), PATH)
