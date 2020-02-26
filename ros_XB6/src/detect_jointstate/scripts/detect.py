#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool
import numpy as np
import math
from scipy.linalg import solve
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import dsntnn
import rpe, rpe_angle

class Predict_Module():
	def __init__(self):
		means, stdevs = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
		self.norm_trans = transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize(mean=means, std=stdevs)
					])

		self.net = rpe.RobotJointModel()
		self.net.load_state_dict(torch.load('joint_model_net_DSNT_100_123_1e-5.pth'))
		self.net.eval()

	def predict(self,data):
		img_1 = Image.open('camera1.png').convert('RGB')
		img_1 = self.norm_trans(img_1)
		img_2 = Image.open('camera2.png').convert('RGB')
		img_2 = self.norm_trans(img_2)
		img_3 = Image.open('camera3.png').convert('RGB')
		img_3 = self.norm_trans(img_3)
		img_1 = img_1.unsqueeze(0)
		heatmap_1 = self.net(img_1)
		c_1 = dsntnn.dsnt(heatmap_1)
		img_2 = img_2.unsqueeze(0)
		heatmap_2 = self.net(img_2)
		c_2 = dsntnn.dsnt(heatmap_2)
		img_3 = img_3.unsqueeze(0)
		heatmap_3 = self.net(img_3)
		c_3 = dsntnn.dsnt(heatmap_3)

		c_1 = ((c_1+1)*224-1)/2
		c_2 = ((c_2+1)*224-1)/2
		c_3 = ((c_3+1)*224-1)/2

		a1, a2, a3, a4, a5 = rpe_angle.model_predict2angle(c_1.detach().numpy()[0], c_2.detach().numpy()[0], c_3.detach().numpy()[0])
		with open("detect_result.txt","w") as f:
			f.write(str(a1)+" ")
			f.write(str(a2)+" ")
			f.write(str(a3)+" ")
			f.write(str(a4[0])+" "+str(a4[1])+" ")
			f.write(str(a5[0])+" "+str(a5[1])+" ")
	    
	def listener(self):
		rospy.init_node('detect_jointstate', anonymous=True)
		rospy.Subscriber("/XB6/detect_command_msg", Bool, self.predict)
		rospy.spin()


if __name__ == '__main__':
	p = Predict_Module()
	p.listener()
