import cv2
import numpy as np

np.set_printoptions(threshold=100000,linewidth=100000)

img_h,img_w = 224,224

for camera_index in range(1,4):
    for data_index in range(12000):
        print(camera_index,"-",data_index)
        depth_img = cv2.imread("./camera_data/camera"+str(camera_index)+"-depth-"+str(data_index)+".png",cv2.IMREAD_GRAYSCALE)
        rgb_img = cv2.imread("./camera_data/camera"+str(camera_index)+"-rgb-"+str(data_index)+".png")
        img = np.zeros((img_h,img_w,3))
        for i in range(img_h):
            for j in range(img_w):
                if(depth_img[i][j] != 0):
                    img[i][j] = rgb_img[i][j]
        cv2.imwrite("./camera_train_data/camera"+str(camera_index)+"-"+str(data_index)+".png", img)