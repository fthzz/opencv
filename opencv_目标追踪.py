import cv2 as cv
import numpy as np
import time
import argparse
OPENCV_tracker={
    'boosting': cv.legacy.TrackerBoosting_create,
    'csrt': cv.legacy.TrackerCSRT_create,
    'kcf': cv.legacy.TrackerKCF_create,
    'mil': cv.legacy.TrackerMIL_create,
    'tld': cv.legacy.TrackerTLD_create,
    'medianflow': cv.legacy.TrackerMedianFlow_create,
    'mosse': cv.legacy.TrackerMOSSE_create
}
trackers=cv.legacy.MultiTracker_create()
cap=cv.VideoCapture("d:\\video2.mp4")

if cap.isOpened():
    while True:
        ret,frame=cap.read()
        if not ret:break
        #追踪目标
        success,boxes=trackers.update(frame)
        print(boxes)
        for box in boxes:
            #box是一个浮点型，画图需要整形且box是一个元组
            (x,y,w,h)=[int(v) for v in box]
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv.imshow('video', frame)
        key=cv.waitKey(10)
        if key==ord('s'):#框选
            #roi返回的是x，y，w，h    /roi也是一个元组
            roi=cv.selectROI('frame',frame,showCrosshair=True,fromCenter=False)#第三个参数显示十字，第四个参数从中间开始框选
            print(roi)
            #创建实际目标跟踪器
            tracker=OPENCV_tracker['kcf']()#创建字典的时候没加括号，加了括号表示创建
            #添加目标追踪器
            trackers.add(tracker,frame,roi)
            print(1)
        elif key==27:#27就是esc
            break
else:
    print('error')
cap.release()
cv.destroyWindow()