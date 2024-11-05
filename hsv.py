import cv2
import numpy as np

img=cv2.imread('/home/fthzzz/Desktop/yolov8/ultralytics-robotdog/y.png')
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower_yellow=np.array([26,43,46])
upper_yellow=np.array([34,255,255])
maskyellow=cv2.inRange(img_hsv,lower_yellow,upper_yellow)

lower_red = np.array([0, 43, 46])
upper_red = np.array([10, 255, 255])
mask0 = cv2.inRange(img,lower_red,upper_red)
lower_red = np.array([156, 43, 46])
upper_red = np.array([180, 255, 255])
mask1 = cv2.inRange(img,lower_red,upper_red)
maskred = mask0 + mask1

cv2.imshow('x',maskred)
cv2.imshow('y',maskyellow)
cv2.waitKey(0)