import cv2
import numpy as np


#任意角度旋转
img=cv2.imread("/home/fthzzz/Desktop/txt/4949.jpg")
h, w, c = img.shape
rotate=90
M = cv2.getRotationMatrix2D((w // 2, h // 2), rotate, 1)
img_ro = cv2.warpAffine(img, M, (w, h))

#cv2.ROTATE_90_COUNTERCLOCKWISE 逆时针90
#cv2.ROTATE_90_CLOCKWISE 顺 90
#cv2.ROTATE_180
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)