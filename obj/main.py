# -*- coding: utf-8 -*-

import module
import cv2
import matplotlib.pyplot as plt
import datetime
import threading
import sys
import os
import numpy as np



dist_list = []
#フレーム数保存用リスト
ran_list = []

todaydetail = datetime.datetime.today()
todaydetail = str(todaydetail.year) + str(todaydetail.month) + str(todaydetail.day) + str(todaydetail.hour) + str(todaydetail.minute) + str(todaydetail.second)

os.makedirs("log", exist_ok=True)
filename= "log/MousePos_" + todaydetail + ".txt"
print("log file = ",filename)
f = open(filename,'w') #データ保存用ファイル

f.write("#")
f.write("FrameNumber")
f.write(" | ")
f.write("Time")
f.write(" | ")
f.write("(X,Y)")
f.write(" | ")
f.write("wake/sleep")
f.write("\n")



instance = module.Tracking_Module()

cap = cv2.VideoCapture("1.mov")

userofst = 0
fps = 0

userofst,fps  = instance.show_video_prev(cap)

print(userofst,fps)

instance.change_video_offset(cap,userofst - 2000)




################################example######################################
frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

showframe  = cap.read()[1]
frame3 = cv2.cvtColor(showframe, cv2.COLOR_RGB2GRAY)

old_x = 0
old_y = 0
frame_number = 0
while(frame_number < fps*4):
    
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




        f.write(str(frame_number))
        f.write(" | ")
        f.write("null")
        f.write(" | ")
        f.write(str(x))
        f.write(",")
        f.write(str(y))
        f.write(" | ")
        f.write("wake")
        f.write("\n")
        
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


f.close()
plt.show()

cap.release()
cv2.destroyAllWindows()

############################################################################
