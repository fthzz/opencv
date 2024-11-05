import cv2
import numpy as np

img=cv2.imread("/home/fthzzz/Desktop/txt/4949.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化，将黑色区域设为255，其他区域设为0
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#findcontours是在黑色下找白色 这个要注意
#必要时可以使用CHAIN_APPROX_NONE_INV 反转二值化
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    #计算周长
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    corner_number = len(approx)
    #框选图形最小外接矩形
    point = cv2.minAreaRect(contour)
    #计算面积
    area = cv2.contourArea(contour)
    if corner_number == 4 :
        print('矩形')