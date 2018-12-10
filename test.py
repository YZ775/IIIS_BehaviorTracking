# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import threading

sleepcnt = 0  #タイマー割り込み時に動いていない場合をカウントしていく
ContinueFlag = 1 #タイマー割り込みを続けるかどうかのフラグ

def IsSleep():
    global sleepcnt
    global ContinueFlag

    if(ContinueFlag == 0): #割り込み終了
        return 1
    if(flag_old == 1 and flag_new == 1): #前回動いてなくて今回も動いてない場合 カウントプラス
        sleepcnt = sleepcnt + 1
        print(sleepcnt)
    if(ContinueFlag == 1): #割り込みを続ける
        t=threading.Timer(0.1,IsSleep)
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
    dist_list = []
    ran_list = []


    todaydetail = datetime.datetime.today()
    todaydetail = str(todaydetail.year) + str(todaydetail.month) + str(todaydetail.day) + str(todaydetail.hour) + str(todaydetail.minute) + str(todaydetail.second)
    filename= "MousePos_" + todaydetail + ".txt"
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
        cap = cv2.VideoCapture(p) #動画読み込み 動画の名前
    else:
        print("input is illiegal")

    try:
        # フレームを3枚取得してグレースケール変換
        frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
        frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
        h, w = frame1.shape
        plt.plot(w,-1*h,'w',marker='.') ##画像の右端の点を白でプロットすることでグラフの描画エリアと画像サイズを合わせる
    except:
        _ = 0





    frame_number = 0
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
        showframe = frame2.copy()

        if(sleepcnt == 10): #寝たら
            plt.subplot(2,1,1)
            plt.plot(x,-1*y,'b',marker='.',markersize=5)#青丸をプロット

        if area > 400: #面積が閾値より大きければ、重心の座標を更新
            sleepcnt = 0 #眠ってるカウントをリセット
            mu = cv2.moments(mask, False)
            try:
                x,y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
                #移動量計算
                distance = (x-before_x)**2+(y-before_y)**2
                if flag_moment == 1:
                    distance = 0
                    flag_moment = 0
                dist_list.append(distance)
                ran_list.append(frame_number)
                plt.plot(x,-1*y,'r',marker='.',markersize=3) #サンプリング点をプロット
                plt.plot([before_x,x],[-1*before_y,-1*y],'g',linewidth = 0.5) #線をプロット
                before_x = x
                before_y = y
                cv2.circle(showframe, (x,y), 3, (255,255,0),-1)
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
                print(flag_old,flag_new)


            except:
                _ = 0

        else :   #面積が閾値より小さければ、前回の座標を表示
            cv2.circle(showframe, (before_x,before_y), 3, (255,255,0),-1)
            f.write(str(frame_number))
            f.write(" | ")
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
            print(flag_old,flag_new)


        # 結果を表示
        cv2.imshow("Frame2", showframe)
        #cv2.imshow("Mask", mask)

        try:
        # 3枚のフレームを更新
            frame1 = frame2
            frame2 = frame3
            frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
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
    plt.subplot(2,1,2)
    plt.plot(ran_list, dist_list)
    plt.show()





if __name__ == '__main__':
    t=threading.Thread(target = IsSleep)
    t.start()
    main()
