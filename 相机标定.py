import cv2 as cv
import cv2
import numpy as np
import glob
"""
相机标定的目的：
机标定的目的有两个，
一个就是矫正由于镜头畸变造成的图片的变形，进行相机标定后可以对这种情况进行校正；
另一个是根据获得得图像重构三维场景，因为标定的过程就是通过一系列的三维点和它对应的二维图像点进行数学变换，求出相机的内参数和外参数。
"""


"""
注意事项：
1.标定图片的数量应该在10~25张之间，图像数量太少，容易导致标定参数不准确。
2.由于一开始我们在代码中给定了棋盘格的模板规格，给定了不同方向上的内角点个数。虽然在拍摄图像需要有一定的角度变化，但不能将x方向和y方向完全颠倒，会导致角点检测错误。
"""
class camera_Calibration():
    def __init__(self):
        pass
    """
    世界坐标系 相机坐标系 图像坐标系
    第一步是从世界坐标系转换为相机坐标系，这一步是三维点到三维点的转换，包括 R,t(相机外参）等参数；
    第二步是从相机坐标系转为图像坐标系，这一步是三维点到二维点的转换，包括 K（相机内参）等参数；
    
    径向畸变（枕型畸变、桶型畸变）和切向畸变（透镜制造上的缺陷）
    """

    """
    可知畸变有 k1 k2 k3 p1 p2五个参数，对于质量比较好的相机来说，切向畸变很小，可忽略
    径向畸变系数k3也可忽略，只计算k1,k2两个参数。张正友标定中就默认为p1 p2为0
    """
    def process(self):
        #用于控制print的数值的精度 方便更好的看到结果
        np.set_printoptions(suppress=True)

        """---------------------------------------
        机器学习模型可以将训练集中的数据划分为若干个组，每个组被称为一个“簇（cluster）”。这种学习方式被称为“聚类（clusting）”
        它的重要特点是在学习过程中不需要用标签对训练样本进行标注。即学习过程能够根据现有训练集自动完成分类（聚类）。
        -------------------------------------------"""

        """---------------------------------------
        k均值聚类：
        将输入数据划分为k个簇的简单的聚类算法，该算法不断提取当前分类的中心点（也称为质心或重心），并最终在分类稳定时完成聚类。
        基本步骤：
            随机选取k个点作为分类的中心点。
            将每个数据点放到距离它最近的中心点所在的类中。
            重新计算各个分类的数据点的平均值，将该平均值作为新的分类中心点。
            重复步骤2和步骤3，直到分类稳定。
            
        注：距离最近: 要进行某种形式的距离计算。（在具体实现时，可以根据需要采用不同形式的距离度量方法。）
        
        它本质上是一个迭代算法，所以会有一个迭代终止条件，即下文中用的criteria
        --------------------------------------------"""
        # 阈值
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        """
        criteria：算法迭代的终止条件。当达到最大循环数目或者指定的精度阈值时，算法停止继续分类迭代计算。该参数由3个子参数构成，分别为type、max_iter和eps。

        type表示终止的类型，可以是三种情况
            cv2.TERM_CRITERIA_EPS：精度满足eps时，停止迭代。
            cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过阈值max_iter时，停止迭代。
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER：上述两个条件中的任意一个满足时，停止迭代。
        max_iter：最大迭代次数。
        eps：精确度的阈值。

        """
        # 棋盘格模板规格
        w = 6  # 内角点个数，内角点是和其他格子连着的点
        h = 4

        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
        objp = np.zeros((w * h, 3), np.float32) # w*h行 3列的矩阵
        # print(objp)

        # .T表示行列置换
        # .reshape（m，n）表示转变为m行n列  其中-1的意思是多少行系统计算，我需要2列
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        # print(objp)

        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = []  # 在世界坐标系中的三维点
        imgpoints = []  # 在图像平面的二维点

        #用于查找文件和目录
        #glob.glob() 表示返回所有匹配条件的文件的路径 返回一个list
        #*.jpg  *表示匹配多个字符
        images = glob.glob('p2/*.jpg')
        i = 0
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 找到棋盘格角点
            # 棋盘图像(8位灰度或彩色图像)  棋盘尺寸  存放角点的位置

            #这是一个棋盘格角点检测算法
            #输入图像（8位灰度图或彩图） 内角点规格 储存角点坐标的数组 标志位（有默认值）
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

            # 如果找到足够点对，将其存储起来
            if ret == True:
                i += 1
                """
                传统角点检测中角点的坐标往往是没法与像素点完全重合的，这是因为一个角点往往有多层像素点组成
                所以为了得到更精准的角点坐标，有了cornersubpix()
                它的作用是求出更精准的角点坐标，精度达到亚像素级别，称为亚像素角点
                """
                # 角点精确检测
                # 输入图像 角点初始坐标 搜索窗口为2*winsize+1 死区 求角点的迭代终止条件
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
                # 将角点在图像上显示
                cv2.drawChessboardCorners(img, (w, h), corners, ret)
                cv2.imshow('findCorners', img)
                cv2.imwrite('p2/h' + str(i) + '.jpg', img)
                cv2.waitKey(10)
        cv2.destroyAllWindows()

        # 标定、去畸变
        # 输入：世界坐标系里的位置 像素坐标 图像的像素尺寸大小 3*3矩阵，相机内参数矩阵 畸变矩阵
        # 输出：标定结果 相机的内参数矩阵 畸变系数 旋转矩阵 平移向量
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        """
        重投影误差：指的真实三维空间点在图像平面上的投影（也就是图像上的像素点）和重投影（其实是用我们的计算值得到的虚拟的像素点）的差值。
        """
        # ret 重投影误差 重投影误差定义为一个特征点在归一化相机坐标系下的估计值与观测值的差，网上均说重投影误差小于0.5 就算效果良好
        # mtx：内参数矩阵
        # dist：畸变系数
        # rvecs：旋转向量 （外参数）
        # tvecs ：平移向量 （外参数）

        print(("ret:"), ret)
        print(("mtx:\n"), mtx)  # 内参数矩阵
        print(("dist:\n"), dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
        print(("rvecs:\n"), rvecs)  # 旋转向量  # 外参数
        print(("tvecs:\n"), tvecs)  # 平移向量  # 外参数

        # 去畸变
        img2 = cv2.imread('p2/1.jpg')
        h, w = img2.shape[:2]
        """
        我们已经得到了相机内参和畸变系数，在将图像去畸变之前，
        我们还可以使用cv.getOptimalNewCameraMatrix()优化内参数和畸变系数，
        通过设定自由自由比例因子alpha:
            当alpha设为0的时候，
            将会返回一个剪裁过的将去畸变后不想要的像素去掉的内参数和畸变系数；
            
            当alpha设为1的时候，将会返回一个包含额外黑色像素点的内参数和畸变系数，并返回一个ROI用于将其剪裁掉
        """
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # 自由比例参数

        #校正后的图像
        #畸变矫正函数
        dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
        # 根据前面ROI区域裁剪图片
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]

        # cv2.imwrite('p2/calibresult.jpg', dst)

        # 反投影误差
        # 通过反投影误差，我们可以来评估结果的好坏。越接近0，说明结果越理想。
        # 通过之前计算的内参数矩阵、畸变系数、旋转矩阵和平移向量，使用cv2.projectPoints()计算三维点到二维图像的投影，
        # 然后计算反投影得到的点与图像上检测到的点的误差，最后计算一个对于所有标定图像的平均误差，这个值就是反投影误差。
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        print(("total error: "), total_error / len(objpoints))


if __name__=='__main__':
    ce=camera_Calibration()
    ce.process()
