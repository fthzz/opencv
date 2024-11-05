import math

"""
这里很重要的一点，对于图片来说，原点是（0，0），不是图像正中心，所以要做向量的话，对数据需要处理
"""


def vector_2d_angle(self):
    '''
        求解二维向量的角度
    '''
    v1_x = self.middle[0] - self.center[0]
    v1_y = self.middle[1] - self.center[1]

    v2_x = self.rect[0] - self.center[0]
    v2_y = self.rect[1] - self.center[1]
    try:
        angle_ = math.degrees(math.acos(
            (v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle_ = 65535.

    #叉乘 可以判断方向 结果有正负

    if v1_x * v2_y - v2_x * v1_y > 0:
        self.direction = -1
    else:
        self.direction = 1

    return angle_