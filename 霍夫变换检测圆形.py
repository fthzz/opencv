import cv2
import numpy as np

class HoughCircles:
    def __init__(self,frame):
        # self.cap=cv2.VideoCapture(0)
        self.frame=frame
        self.bbox = None
        self.image_width = self.frame.shape[0]
        self.image_height = self.frame.shape[1]
        self.min_area=500

    def detect(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # 使用高斯滤波平滑图像
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 使用霍夫变换检测圆形
        #circles返回一个三维数组 每组表示一个圆，分别是中心点x，y，r半径
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 100)

        # 如果检测到圆形，绘制边界
        if circles is not None:
            # 将圆形坐标转换为整数
            circles = np.round(circles[0, :]).astype("int")

            # 定义一个阈值，表示两个圆心之间的最大距离，用于判断是否合并
            threshold = 10
            # 定义一个列表，用于存储合并后的圆形
            merged_circles = []
            # 遍历每个圆形
            for (x1, y1, r1) in circles:
                # 定义一个标志，表示当前的圆形是否已经被合并过
                merged = False
                # 遍历已经合并过的圆形列表
                for i in range(len(merged_circles)):
                    # 获取已经合并过的圆形的坐标和半径
                    (x2, y2, r2) = merged_circles[i]
                    # 计算两个圆心之间的距离
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    # 如果距离小于阈值，说明两个圆形可以合并
                    if distance < threshold:
                        # 将当前的圆形和已经合并过的圆形进行平均，得到新的圆形
                        if r1 >= r2:
                            merged_circles[i] = (x1, y1, r1)
                        else:
                            merged_circles[i] = (x2, y2, r2)
                        # 设置标志为True，表示当前的圆形已经被合并过
                        merged = True
                        # 跳出循环，不再遍历其他已经合并过的圆形
                        break
                # 如果当前的圆形没有被合并过，就将它添加到已经合并过的圆形列表中
                if not merged:
                    merged_circles.append((x1, y1, r1))

            # 遍历已经合并过的圆形列表，找到最大的圆，作为检测结果
            biggest = None
            biggest_r = -1
            for (x, y, r) in merged_circles:
                if r > biggest_r:
                    biggest = (x, y, r)
                    biggest_r = r

            # 将圆转换成bbox
            #bbox 全称叫bounding box 边界框
            if biggest is not None:
                self.bbox = np.array([
                    [biggest[0] - biggest[2], biggest[1] - biggest[2]],
                    [biggest[0] + biggest[2], biggest[1] + biggest[2]]], np.int32)
                return self.__refine_bbox(self.bbox)
        return None


    def __refine_bbox(self, bbox):
        # 将array中的数字框定在a_min和a_max之间，大于或小于直接取边界值
        bbox[:, 0] = np.clip(bbox[:, 0], 0, self.image_width)
        bbox[:, 1] = np.clip(bbox[:, 1], 0, self.image_height)
        w, h = bbox[1] - bbox[0]
        if w <= 0 or h <= 0 or w * h <= self.min_area:
            return None
        else:
            return bbox

    def img_exchange(self):
        if self.bbox is not None:
            image = self.frame[self.bbox[0][1]:self.bbox[1][1], self.bbox[0][0]:self.bbox[1][0], :]
            return image
        return self.frame