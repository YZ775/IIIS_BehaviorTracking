# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import threading
import sys
import os

sleepcnt = 0  #タイマー割り込み時に動いていない場合をカウントしていく
ContinueFlag = 1 #タイマー割り込みを続けるかどうかのフラグ


def IsSleep():
    global sleepcnt
    global ContinueFlag

    if(ContinueFlag == 0): #割り込み終了
        return 1
    if(flag_old == 1 and flag_new == 1): #前回動いてなくて今回も動いてない場合 カウントプラス
        sleepcnt = sleepcnt + 1
        #print(sleepcnt)
    if(ContinueFlag == 1): #割り込みを続ける
        t=threading.Timer(0.1,IsSleep)
        t.setDaemon(True)
        t.start()



# フレーム差分の計算
def frame_sub(img1, img2, img3, th):
    # フレームの絶対差分
    diff1 = cv2.absdiff(img1, img2)
    diff2 = cv2.absdiff(img2, img3)

    # 2つの差分画像の論理積
    diff = cv2.bitwise_xor(diff1, diff2)

    # 二値化処理
    diff[diff < th] = 0
    diff[diff >= th] = 255

    # メディアンフィルタ処理（ゴマ塩ノイズ除去）
    mask = cv2.medianBlur(diff, 3)

    return  mask



ofst = 0
def ShowPrev(path):
    global ofst
    def ChangeBar(val):
        global ofst
        ofst = val
        cap_prev.set(0,val)
        frame = cap_prev.read()[1]
        try:
            cv2.imshow("prev",frame)
        except:
            print("end of video")

    cap_prev = cv2.VideoCapture(path)
    video_frame = cap_prev.get(cv2.CAP_PROP_FRAME_COUNT) # フレーム数を取得する
    video_fps = cap_prev.get(cv2.CAP_PROP_FPS)           # FPS を取得する
    video_len_sec = int((video_frame / video_fps)*1000) #長さ[ms]を取得

    cap_prev.set(0,0*1000)
    cv2.namedWindow('prev')
    cv2.createTrackbar("Frame", "prev", 0, video_len_sec, ChangeBar)
    frame = cap_prev.read()[1]
    cv2.imshow("prev",frame)
    
    while(cap_prev.isOpened()):
        key = cv2.waitKey(1)&0xff
        if key == ord('d'):
            ChangeBar(ofst+100)
            cv2.setTrackbarPos("Frame","prev", ofst) 
            print(ofst)
        
        if key == ord('a'):
            ChangeBar(ofst-100)
            cv2.setTrackbarPos("Frame","prev", ofst)
            print(ofst)

        if key == ord('w'):
            ChangeBar(ofst+10)
            cv2.setTrackbarPos("Frame","prev", ofst) 
            print(ofst)
        
        if key == ord('s'):
            ChangeBar(ofst-10)
            cv2.setTrackbarPos("Frame","prev", ofst)
            print(ofst)
            
        if key == ord('q'):
            break
    
    cap_prev.release()
    cv2.destroyAllWindows()
    return ofst

#####################################
#前回の座標値を記憶する変数
before_x = 0
before_y = 0
x = 0
y = 0
flag_old = 0
flag_new = 0


def main():

    global before_x #グローバル変数であることを宣言
    global before_y
    global x
    global y
    global flag_old
    global flag_new
    global sleepcnt
    global ContinueFlag
    #移動量保存用リスト
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

    print("select Source")
    select = input("0:from camera\n1:from video\n>> ")

    if(select == "0"):
        cap = cv2.VideoCapture(0)  # カメラのキャプチャ

    if(select == "1"):
        p = input("enter video path\n>> ")
        isShift = input("change offset? (0/1)\n>>") #動画スタートのオフセット
        if(isShift == "1"): #数字でない場合強制終了
            print("########Operation Key############")
            print("w: +10F a:-100F s:-10F d:+100F")
            shift_time = ShowPrev(p)
            cap = cv2.VideoCapture(p) #動画読み込み 動画の名前
            print(shift_time)
            cap.set(0,shift_time)
        else:
            cap = cv2.VideoCapture(p) #動画読み込み 動画の名前
           
        
       
    else:
        print("illegal input")

    try:
        # フレームを3枚取得してグレースケール変換
        frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
        
        nextframe = cap.read()[1] #再生用に1枚抜く

        frame3 = cv2.cvtColor(nextframe, cv2.COLOR_RGB2GRAY)
        h, w = frame1.shape
        plt.plot(w,-1*h,'w',marker='.') ##画像の右端の点を白でプロットすることでグラフの描画エリアと画像サイズを合わせる
    except:
        _ = 0





    frame_number = 0
    #移動量用のフラッグ設定
    flag_moment = 1
    while(cap.isOpened()):
        time_stamp = datetime.datetime.now()
        frame_number += 1
        # フレーム間差分を計算
        mask = frame_sub(frame1, frame2, frame3, th=10)
        area = cv2.countNonZero(mask)
        """
        #輪郭検出
        image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]

        #マスクの面積計算
        area = cv2.contourArea(cnt)
        """
        #print(area)
        showframe = nextframe.copy()
        

        if(sleepcnt == 10): #寝たら
            #サブプロットに変更
            plt.subplot(2,1,1)
            plt.plot(x,-1*y,'b',marker='.',markersize=5)#青丸をプロット
            cv2.circle(showframe, (x,y), 15, (0,255,0),-1)

        
        if area > 400: #面積が閾値より大きければ、重心の座標を更新
            sleepcnt = 0 #眠ってるカウントをリセット
            mu = cv2.moments(mask, False)
            try:
                x,y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
                #移動量計算
                distance = (x-before_x)**2+(y-before_y)**2
                #before_x, before_yが設定されていない状態（一番初め）
                if flag_moment == 1:
                    distance = 0
                    #フラッグ消す
                    flag_moment = 0
                #リストにデータ追加
                dist_list.append(distance)
                ran_list.append(frame_number)
                plt.subplot(2,1,1)
                plt.plot(x,-1*y,'r',marker='.',markersize=3) #サンプリング点をプロット
                plt.subplot(2,1,1)
                plt.plot([before_x,x],[-1*before_y,-1*y],'g',linewidth = 0.5) #線をプロット
                before_x = x
                before_y = y
                cv2.circle(showframe, (x,y), 7, (0,0,255),-1)
                f.write(str(frame_number))
                f.write(" | ")
                if (select == "1"):#ビデオ読み込みの場合，タイムスタンプはnull
                    f.write("null")
                else :
                    f.write(str(time_stamp))
                f.write(" | ")
                f.write(str(x))
                f.write(",")
                f.write(str(y))
                f.write(" | ")
                f.write("wake")
                f.write("\n")

                flag_old = flag_new #動いてるかどうかフラグ更新
                flag_new = 0
                #print(flag_old,flag_new)


            except:
                _ = 0

        else :   #面積が閾値より小さければ、前回の座標を表示
            cv2.circle(showframe, (before_x,before_y), 7, (255,0,0),-1)
            f.write(str(frame_number))
            f.write(" | ")
            #リストにデータ追加
            distance = 0
            dist_list.append(distance)
            ran_list.append(frame_number)
            if (select == "1"): #ビデオ読み込みの場合，タイムスタンプはnull
                f.write("null")

            else:
                f.write(str(time_stamp))

            f.write(" | ")
            f.write(str(before_x))
            f.write(",")
            f.write(str(before_y))
            f.write(" | ")
            if(sleepcnt >= 10): #寝てたら
                f.write("sleep")
            else:
                f.write("wake")

            f.write("\n")

            flag_old = flag_new #動いてるかどうかフラグ更新
            flag_new = 1
            #print(flag_old,flag_new)

        # 結果を表示
        cv2.imshow("showframe", showframe)
        #cv2.imshow("Mask", mask)
        
        try:
        # 3枚のフレームを更新
            frame1 = frame2
            frame2 = frame3
            nextframe = cap.read()[1]
            frame3 = cv2.cvtColor(nextframe, cv2.COLOR_BGR2GRAY)
        except:
            break

        # qキーが押されたら途中終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    f.close()
    ContinueFlag = 0
    #t.cancel()
    cap.release()
    cv2.destroyAllWindows()
    #サブプロットで移動量を描画　横軸はフレーム数
    plt.subplot(2,1,2)
    plt.plot(ran_list, dist_list)
    plt.show()





if __name__ == '__main__':
    t=threading.Thread(target = IsSleep)
    t.setDaemon(True)
    t.start()
    main()
