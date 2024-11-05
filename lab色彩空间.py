import cv2 as cv
import numpy as np

#lab色彩空间标准划分
# l[0,100]
# a,b[-127,127]
# 但使用cvt转换出来都是[0，256]

img=cv.imread("/home/fthzzz/Desktop/yolov8/ecar-vision/apple/apple_frame/apple (143).jpg",cv.IMREAD_UNCHANGED)
img_lab=img*(1./255)
img_lab=cv.cvtColor(np.float32(img_lab),cv.COLOR_BGR2Lab)
h,w=img_lab.shape[:2]
for i in range (h):
    for j in range (w):
        img_lab[i][j][0]*=2.5
# print(img_lab[:,:,0])
img_ex=cv.cvtColor(img_lab,cv.COLOR_Lab2BGR)
img_ex/=(1./255)
img_ex=img_ex.astype(np.uint8)
cv.imshow('x',img_lab)
cv.waitKey(0)
cv.destroyWindow()
