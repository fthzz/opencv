import cv2 as cv
import numpy as np
img=cv.imread("/home/fthzzz/Desktop/ecar-vision/beam.png",cv.IMREAD_UNCHANGED)

#图像卷积
ker=np.ones((5,5),np.float32)/(5*5)
img_fliter=cv.filter2D(img,-1,ker)

#低通滤波 去噪平滑

#均值滤波
img_bulr=cv.blur(img,(3,3))

#高斯滤波
img_Gussia=cv.GaussianBlur(img,(3,3),0)

#中值滤波
img_media=cv.medianBlur(img,3)

#双边滤波 去噪保边 时间较长
#后面两个参数越大 范围越大的空间颜色会相互影响同化
img_filter=cv.bilateralFilter(img,5,20,50)


#高通滤波 突出边缘
#也可以认为是边缘检测
# cv.Sobel()
# cv.Scharr()
# cv.Laplacian()
# cv.Canny()

# cv.imshow('1',img_bulr)
# cv.imshow('2',img_Gussia)
# cv.imshow('3',img_media)
# cv.waitKey(0)
# cv.destroyWindow()