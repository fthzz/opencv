import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while (cap.isOpened()):
    ret,frame=cap.read()
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    top_size, bottom_size, left_size, right_size = (15, 15, 15, 15)
    #以特定值给图片填充一个边框
    img = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=(125,62,25))

    lower=np.array([20,45,90])
    upper=np.array([25,120,150])
    mask=cv2.inRange(img,lower,upper)

    #中值滤波
    blur=cv2.medianBlur(mask,5)
    #膨胀
    kernel=np.ones((5,5),np.uint8)
    blur=cv2.dilate(blur,kernel,iterations=3)

    blur=cv2.Canny(blur,1,220)
    contours,hierarchy = cv2.findContours(blur,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area=cv2.contourArea(contour)
        if (area>7500):
            x,y,w,h=cv2.boundingRect(contour)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

    cv2.imshow('1',frame)

    if cv2.waitKey(20) &0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()