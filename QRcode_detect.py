import cv2
import numpy as np
import math

class QRcode_Detect:
    # def __init__(self):
    #    self.is_find=0
    def isQRcode_point(self,con1, con2, con3):
        base1 = cv2.contourArea(con1) / 49
        base2 = cv2.contourArea(con2) / 25
        base3 = cv2.contourArea(con3) / 9
        base = min(base1, base3, base2)
        if base1 - base < base / 4 and base2 - base <= base / 4 and base3 - base <= base / 4:
            return True
        return False

    def qrcode_point_sort(self,points):
        p1, p2, p3 = points
        v12 = [p2[0] - p1[0], p2[1] - p1[1]]
        v13 = [p3[0] - p1[0], p3[1] - p1[1]]

        def get_angle(v1, v2):
            return math.acos(
                (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1]))
            ) / math.pi * 180.0

        def r_in(value, target=90, miss=10):
            if value > target + miss or value < target - miss:
                return False
            return True

        if r_in(get_angle(v12, v13), 90):
            if (v12[1] < 0 and v13[0] > 0) or (v12[0] < 0 and v13[1] < 0) or (v12[0] > 0 and v13[1] > 0) or (
                    v12[1] > 0 and v12[0] < 0):
                return [p1, p2, p3]
            elif (v12[0] > 0 and v13[1] < 0) or (v12[1] < 0 and v13[0] < 0) or (v12[1] > 0 and v13[0] > 0) or (
                    v12[0] < 0 and v13[1] > 0):
                return [p1, p3, p2]
        elif r_in(get_angle(v12, v13), 45):
            if math.hypot(v12[0], v12[1]) < math.hypot(v13[0], v13[1]):
                if np.cross(np.array(v12 + [0]), np.array(v13 + [0]))[2] > 0:
                    return [p2, p3, p1]
                else:
                    return [p2, p1, p3]
            else:
                if np.cross(np.array(v12 + [0]), np.array(v13 + [0]))[2] > 0:
                    return [p3, p1, p2]
                else:
                    return [p3, p2, p1]
        else:
            print("ERROR")
            return False

    def qrcode_output(self,img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        level = 0
        contours_sign = []
        count = 0
        for i in range(len(hierarchy[0])):
            hie = hierarchy[0][i]
            if hie[2] != -1:
                level += 1
            elif hie[2] == -1 and level != 0:
                level += 1
                if level == 3:
                    count += 1
                    # print("找到第",count,"个三层的")
                    if self.isQRcode_point(contours[i - 2], contours[i - 1], contours[i]):
                        contours_sign.append(contours[i])
                level = 0
            else:
                level = 0

        sign_point_center = []
        for contour in contours_sign:
            tmpcont = contour[:, 0]
            avg_x, avg_y = 0, 0
            for tmp_x, tmp_y in tmpcont:
                avg_x += tmp_x
                avg_y += tmp_y
            avg_x /= contour.shape[0]
            avg_y /= contour.shape[0]
            sign_point_center.append([avg_x, avg_y])

        # 获取点的位置
        sign_point_center = self.qrcode_point_sort(sign_point_center)

        # 对图片进行仿射变换 将倾斜二维码摆正
        min_x, max_x = min([x[0] for x in sign_point_center]), max([x[0] for x in sign_point_center])
        min_y, max_y = min([x[1] for x in sign_point_center]), max([x[1] for x in sign_point_center])
        if max_x - min_x > max_y > min_y:
            max_y = min_y + max_x - min_x
        else:
            max_y = min_x + max_y - min_x

        M = cv2.getAffineTransform(
            np.float32([
                [sign_point_center[0][0], sign_point_center[0][1]], [sign_point_center[1][0], sign_point_center[1][1]],
                [sign_point_center[2][0], sign_point_center[2][1]]
            ]),
            np.float32([
                [min_x, min_y], [max_x, min_y], [min_x, max_y]
            ])
        )
        # 输出的正确位置的图片
        qrcode_output = cv2.warpAffine(img, M, (int(img.shape[1] * 1.5), int(img.shape[0])))
        return qrcode_output

    def qrcode_read(self,img):
        x = cv2.QRCodeDetector()
        val, _, _ = x.detectAndDecode(img)
        if val:
            return val
        return
if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)#0表示默认第一个摄像头
    # # cap.set(3,640)
    # # cap.set(4,480)
    # ad = QRcode_Detect()
    # while True:
    #     ret, frame = cap.read()
    #     frame = cv2.rotate(frame, cv2.ROTATE_180)
    #     output_img=ad.qrcode_output(frame)
    #     val=ad.qrcode_read(output_img)
    #     # cv2.imshow("img", frame)
    #     if cv2.waitKey(100) & 0xff == ord('q'):
    #         break
    # cap.release()
    image = cv2.imread("/home/fthzzz/Desktop/ecar-vision/qr_turn.png", cv2.IMREAD_UNCHANGED)
    img = image.copy()
    ad = QRcode_Detect()
    output_img = ad.qrcode_output(img)
    val=ad.qrcode_read(output_img)
    if val:
        print(val)
    else:
        print("error")
    cv2.destroyAllWindows()
