import cv2
cap = cv2.VideoCapture(6)

while True:
    ret, frame = cap.read()
    # cap.set(3, 640)
    # cap.set(4, 480)
    cv2.imshow("img", frame)
    cv2.waitKey(1)
cap.release()
