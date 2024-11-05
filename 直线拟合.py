import cv2
import numpy as np

class FitLine:
    def __init__(self):
        self.contour=None

    def detect(self):
        if self.contour is not None:
            # 得到指针轮廓的最佳拟合直线，并获取其斜率
            vx, vy, x0, y0 = cv2.fitLine(self.contour, cv2.DIST_L2, 0, 0.01, 0.01)
            slope = vy / vx
            # 将斜率转换为角度，这个角度是指针相对于水平线的夹角
            angle = np.arctan(slope) * 180 / np.pi