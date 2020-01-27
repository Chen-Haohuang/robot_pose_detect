import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

img_h, img_w = 224, 224

class RobotJointModel(nn.Module):

	def __init__(self, _means, _stdevs):
		super(RobotJointModel, self).__init__()
		self.features = nn.Sequential(
			torchvision.models.vgg16(pretrained=True).features[:12]
		)





def get_data_input(path):
	src_image_list = os.listdir(path)
	data_num = 10 #len(src_image_list))
	imgs = []
	for i in range(data_num):
		img = cv2.imread(path+'/'+src_image_list[i])
		imgs.append(img)

	imgs = np.array(imgs)#.reshape((data_num, img_h, img_w, 3))
	imgs = imgs.astype(np.float32)/255
	
	return imgs

def get_mean_stdevs(imgs):
	means, stdevs = [], []
	for i in range(3):
		pixels = imgs[:,:,:,i].ravel()  # 拉成一行
		means.append(np.mean(pixels))
		stdevs.append(np.std(pixels))
	means.reverse() # BGR --> RGB
	stdevs.reverse()

	return means, stdevs

imgs = get_data_input('./camera_data_224')
print(imgs.shape)

means, stdevs = get_mean_stdevs(imgs)
print(means, stdevs)

norm_trans = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=means, std=stdevs)
			 ])

for i in range(len(imgs)):
	imgs[i] = norm_trans(imgs[i])

print(imgs.shape)
print(RobotJointModel(means, stdevs))