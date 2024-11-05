import cv2
import numpy as np
"""
流程：
1双目标定，得到内外参
2立体矫正
3立体匹配
4视差计算
5深度计算

"""

"""
第一个概念 视差
csdn
第一幅图是三维世界的一个点在两个相机的成像，我们可以相信的是，这两个在各自相机的相对位置基本不可能是一样的，
而这种位置的差别，也正是我们眼睛区别3D和2D的关键，将右边的摄像机投影到左边，怎么做呢？

因为他的坐标和左边相机的左边相距Tx（标定测出来的外参数），所以它相当于在左边的相机对三维世界内的（x-tx,y,z）进行投影，
所以这时候，一个完美的形似三角形就出来，这里视差就是d=x-x‘，
"""

"""
知道视差后，我们如何将 左边摄像头图片的点 在 右边摄像头 找到对应位置，这个就是

立体匹配：
opencv中提供了很多的立体匹配算法，类似于局部的BM，全局的SGBM等等，这些算法的比较大概是，速度越快的效果越差。
如果不是很追究时效性，并且你的校正做的不是很好的话..推荐使用SGBM

这里我想提一下的是为什么做立体匹配有用，原因就是极线约束，这也是个很重要的概念，
左摄像机上的一个点，对应三维空间上的一个点，当我们要找这个点在右边的投影点时，有必要把这个图像都遍历一边么，当然不用,只需要遍历一条线
"""


class binocular_ranging():
    #第一步 双目标定
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[1499.641, 0, 1097.616],
                                         [0., 1497.989, 772.371],
                                         [0., 0., 1.]])
        # 右相机内参
        self.cam_matrix_right = np.array([[1494.855, 0, 1067.321],
                                          [0., 1491.890, 777.983],
                                          [0., 0., 1.]])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.1103, 0.0789, -0.0004, 0.0017, -0.0095]])
        self.distortion_r = np.array([[-0.1065, 0.0793, -0.0002, -8.9263e-06, -0.0161]])

        # 旋转矩阵
        self.R = np.array([[0.9939, 0.0165, 0.1081],
                           [-0.0157, 0.9998, -0.0084],
                           [-0.1082, 0.0067, 0.9940]])

        # 平移矩阵
        self.T = np.array([[-423.716], [2.561], [21.973]])

        # 主点列坐标的差
        self.doffs = 0.0

        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False

    def setMiddleBurryParams(self):
        self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
                                         [0., 3997.684, 187.5],
                                         [0., 0., 1.]])
        self.cam_matrix_right = np.array([[3997.684, 0, 225.0],
                                          [0., 3997.684, 187.5],
                                          [0., 0., 1.]])
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        self.R = np.identity(3, dtype=np.float64)
        self.T = np.array([[-193.001], [0.0], [0.0]])
        self.doffs = 131.111
        self.isRectified = True


    # 第二步立体矫正
    def getRectifyTransform(self,img1,img2):
        """
        主要利用 “极线矫正”
        目的是将拍摄于同一场景的左右两个视图进行数学上的投影变换，使得两个图像平面共面且平行于基线，简称共面行对准。
        经过这样的校正过程之后，两幅图中的极线就会完全水平，从而导致空间中的同一个点在左右两幅图中的像素位置位于同一行。

        简而言之就是将两个平面投影到同一平面上。

        """
        left_K = self.cam_matrix_left
        right_K = self.cam_matrix_right
        left_distortion = self.distortion_l
        right_distortion = self.distortion_r
        R = self.R
        T = self.T

        # 计算校正变换
        height = int(img1.shape[0])
        width = int(img1.shape[1])

        #立体校正函数
        #内部采用了bouguet的极线校正算法
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                          (width, height), R, T, alpha=0)

        #此函数同时完成极线校正和畸变校正
        map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

        #处理后的图片
        rectifyed_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_AREA)
        rectifyed_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_AREA)

        return rectifyed_img1,rectifyed_img2

    #第三步立体匹配
    """
    常用的立体匹配方法基本上可以分为两类：局部方法（例如，BM、SGM、ELAS、Patch Match等）和非局部的，即全局方法（例如，Dynamic Programming、Graph Cut、Belief Propagation等）。
    局部方法计算量小，但匹配质量相对较低，全局方法省略了代价聚合而采用了优化能量函数的方法，匹配质量较高，但是计算量也比较大。
    目前OpenCV中已经实现的方法有BM、binaryBM、SGBM、binarySGBM、BM(cuda)、Bellief Propogation(cuda)、Constant Space Bellief Propogation(cuda)这几种方法。
    """
    """
    大部分立体匹配算法的计算过程可以分成以下几个阶段：匹配代价计算、代价聚合、视差优化、视差细化。
    立体匹配是立体视觉中一个很难的部分，主要困难在于：
    1.图像中可能存在重复纹理和弱纹理，这些区域很难匹配正确；
    2.由于左右相机的拍摄位置不同，图像中几乎必然存在遮挡区域，在遮挡区域，左图中有一些像素点在右图中并没有对应的点，反之亦然；
    3.左右相机所接收的光照情况不同；
    4.过度曝光区域难以匹配；
    5.倾斜表面、弯曲表面、非朗伯体表面；
    6.较高的图像噪声等。

    """
    #SGBM
    # 视差计算
    def stereoMatchSGBM(self,left_image, right_image, down_scale=False):
        # SGBM匹配参数设置
        if left_image.ndim == 2:
            img_channels = 1
        else:
            img_channels = 3
        blockSize = 3
        paraml = {'minDisparity': 0,
                  'numDisparities': 128,
                  'blockSize': blockSize,
                  'P1': 8 * img_channels * blockSize ** 2,
                  'P2': 32 * img_channels * blockSize ** 2,
                  'disp12MaxDiff': 1,
                  'preFilterCap': 63,
                  'uniquenessRatio': 15,
                  'speckleWindowSize': 100,
                  'speckleRange': 1,
                  'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                  }

        # 构建SGBM对象
        left_matcher = cv2.StereoSGBM_create(**paraml)
        paramr = paraml
        paramr['minDisparity'] = -paraml['numDisparities']
        right_matcher = cv2.StereoSGBM_create(**paramr)

        # 计算视差图
        size = (left_image.shape[1], left_image.shape[0])
        if down_scale == False:
            disparity_left = left_matcher.compute(left_image, right_image)
            disparity_right = right_matcher.compute(right_image, left_image)

        else:
            left_image_down = cv2.pyrDown(left_image)
            right_image_down = cv2.pyrDown(right_image)
            factor = left_image.shape[1] / left_image_down.shape[1]

            disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
            disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
            disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
            disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
            disparity_left = factor * disparity_left
            disparity_right = factor * disparity_right

        # 真实视差（因为SGBM算法得到的视差是×16的）
        trueDisp_left = disparity_left.astype(np.float32) / 16.
        trueDisp_right = disparity_right.astype(np.float32) / 16.

        return trueDisp_left, trueDisp_right