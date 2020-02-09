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
import rpe_angle

use_gpu = torch.cuda.is_available()

batch_size = 16
lr_init = 1e-4
max_epoch = 10

all_data_list = [i for i in range(12000)]
test_data_list = random.sample(all_data_list, 800)
train_data_list = list(set(all_data_list) - set(test_data_list))
random.shuffle(test_data_list)
random.shuffle(train_data_list)

train_data = rpe_angle.MyDataset(train_data_list)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=64)

test_data = rpe_angle.MyDataset(test_data_list)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=64)

net = rpe_angle.RobotAngleModel()
if(use_gpu):
	net = net.cuda()
net.initialize_weights()

criterion = nn.SmoothL1Loss()                                                   # 选择损失函数
if(use_gpu):
	criterion = criterion.cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-5, momentum=0.9, dampening=0.1)    # 选择优化器
# optimizer = torch.optim.Adam(net.parameters(), lr=lr_init)    # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)     # 设置学习率下降策略
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, threshold=1e-7, min_lr=1e-7, factor=0.9)     # 设置学习率下降策略

for epoch in range(max_epoch):
    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    pre_i = -1

    for i, data in enumerate(train_loader):
        inputs, labels, _ = data
        inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)

        if(use_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        if(use_gpu):
            loss = loss.cpu()
        loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 9 or i == len(train_loader)-1:
            loss_avg = loss_sigma / (i-pre_i)
            pre_i = i
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.8f}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg))

    loss_sigma = 0.0
    net.eval()
    for i, data in enumerate(test_loader):

        # 获取图片和标签
        images, labels, name = data
        images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)

        if(use_gpu):
            images, labels = images.cuda(), labels.cuda()

        # forward
        outputs = net(images)
        
        out_for_image = outputs*255
        out_for_image = out_for_image.cpu()
        for b in range(len(out_for_image)):
            joint_image = np.zeros((56,56))
            for j in range(5):
                joint_image += out_for_image.detach().numpy()[b][j]
            cv2.imwrite('./test_predict/'+name[b]+'-joints.png', joint_image)

        # 计算loss
        loss = criterion(outputs, labels.float())
        loss_sigma += loss.item()

    loss_avg = loss_sigma / len(test_loader)
    print("Testing: Epoch[{:0>3}/{:0>3}] Loss: {:.8f}".format(
        epoch + 1, max_epoch, loss_avg))