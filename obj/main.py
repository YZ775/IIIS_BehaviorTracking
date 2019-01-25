# -*- coding: utf-8 -*-

import module
import cv2
import matplotlib.pyplot as plt
import datetime
import threading
import sys
import os



dist_list = []
#フレーム数保存用リスト
ran_list = []

instance = module.Tracking_Module()

cap = cv2.VideoCapture("1.mov")
instance.show_video_prev(cap)


################################example######################################
frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

showframe  = cap.read()[1]
frame3 = cv2.cvtColor(showframe, cv2.COLOR_RGB2GRAY)

old_x = 0
old_y = 0
while(True):
    
    frame_number += 1
    
    try:
        mask, area, x,y = instance.frame_diff(frame1,frame2,frame3,10)
        frame1 = frame2
        frame2 = frame3
        #frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
        showframe  = cap.read()[1]
        frame3 = cv2.cvtColor(showframe, cv2.COLOR_RGB2GRAY)
    except:
        break


    print(x,y)

    if(area > 400):
        distance = np.sqrt((x - old_x)**2+(y - old_y)**2)
        
        plt.subplot(2,1,1)
        plt.plot(x,-1*y,'r',marker='.',markersize=3) #サンプリング点をプロット
        plt.subplot(2,1,1)
        plt.plot([old_x,x],[-1*old_y,-1*y],'k',linewidth = 0.5,alpha = 0.5) #線をプロット
        
        old_x = x
        old_y = y

    cv2.circle(showframe, (old_x,old_y), 15, (0,255,255),-1) #黄丸をカメラ映像に表示

    cv2.imshow('aaa',showframe)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.show()

cap.release()
cv2.destroyAllWindows()

############################################################################
