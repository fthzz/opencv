import cv2
img=cv2.imread('d:\\f3.jpg',cv2.IMREAD_UNCHANGED)#不改变原本图像
b=img[58,99,0]#看一个通道的值
print(b)#查看像素点 0为blue 1为green 2为red
print(img[23,45])#直接打印所有通道的值
img[34,55]=[124,123,122]#可直接赋值
img[23,45,0]=233#可通过不同颜色模式进行赋值

#批量更改   255表示白色
img[100:150,60:90]=[255,255,255]#第一百行到一百五十行
                                                    #第六十列到第九十列     都变成后面那个颜色
import cv2
import numpy as np
img=cv2.imread('d:\\f3.jpg',cv2.IMREAD_UNCHANGED)
img.itemset((100,20,1),222)#修改像素
p=img.item(100,29,0)#读取像素 使用item必须加上通道
print(p)
