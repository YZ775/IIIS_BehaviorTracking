# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import threading
import sys
import os
import re

sleepcnt = 0  #タイマー割り込み時に動いていない場合をカウントしていく
ContinueFlag = 1 #タイマー割り込みを続けるかどうかのフラグ
th_param = 400 #マスクの閾値


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
fps = 0
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
    cv2.createTrackbar("Time[ms]", "prev", 0, video_len_sec, ChangeBar)
    frame = cap_prev.read()[1]
    cv2.imshow("prev",frame)

    while(cap_prev.isOpened()):
        key = cv2.waitKey(1)&0xff
        if key == ord('d'):
            ChangeBar(ofst+100)
            cv2.setTrackbarPos("Time[ms]","prev", ofst)
            print(ofst)

        if key == ord('a'):
            ChangeBar(ofst-100)
            cv2.setTrackbarPos("Time[ms]","prev", ofst)
            print(ofst)

        if key == ord('w'):
            ChangeBar(ofst+10)
            cv2.setTrackbarPos("Time[ms]","prev", ofst)
            print(ofst)

        if key == ord('s'):
            ChangeBar(ofst-10)
            cv2.setTrackbarPos("Time[ms]","prev", ofst)
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

def logger(f_path, day_name, b_s_s, a_s_s, late):
    #ログの処理
    f_before = open(f_path, 'r')

    os.makedirs("log_after", exist_ok=True)
    text = "log_after/after-" + day_name + ".txt"

    f_after  = open(text, 'w')

    l = f_before.readlines()

    #rev_l = reversed(l)
    l.reverse() #リストの反転
    l_flag = 0
    #座標
    lx_bef = 0
    ly_bef = 0

    new_data = []

    for data in l:
        data_s = data.split("|") #リストを分割
        #print(data_s)

        #フリーズしている
        if l_flag == 1:
            #座標値を取得
            data_ss_p = data_s[2]
            data_z_p = data_ss_p.split(",")
            #print("{0} {1} {2} {3}".format(lx_bef, ly_bef, data_z_p[0], data_z_p[1]))

            #前回の座標値と同じ
            if lx_bef == data_z_p[0] and ly_bef == data_z_p[1]:

                text_a = "{}|{}|{}| {}"
                text_b = text_a.format(data_s[0], data_s[1], data_s[2], "freez\n")
                #f.write(text_b) 新しいリストに入力
                new_data.append(text_b)

            #座標値が違う
            else:
                l_flag = 0
                #新しいリストに入力
                new_data.append(data)

        #フリーズしていない
        else:
            if re.search('freez', data_s[3]):
                #フラグを立てる
                l_flag = 1
                #print("check")

                #座標値を保存
                data_ss = data_s[2]
                data_z = data_ss.split(",")
                #print(data_z)
                lx_bef = data_z[0]
                ly_bef = data_z[1]
                #新しいリストに入力
                new_data.append(data)
            #問題ない
            else:
                new_data.append(data)

    new_data.reverse()
    #print(new_data)
    for logging in new_data:
        f_after.write(logging)

    f_after.write("\n")

    f_after.write("Before_Shock Sum of Motion Index: ")
    f_after.write(str(b_s_s))
    f_after.write("\n")

    f_after.write("After_Shock Sum of Motion Index: ")
    f_after.write(str(a_s_s))
    f_after.write("\n")

    f_after.write("A-B/A+B: ")
    f_after.write(str(late))
    f_after.write("\n")



    f_before.close()
    f_after.close()



def main(movie_path):

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

    print("Enter Threshold　parametar")
    th_param = int(input("default:400 \n >>"))
    print("\n")

    select = 1
    p = movie_path

    print("########Operation Key############")
    print("w: +10[ms] a:-100[ms] s:-10[ms] d:+100[ms]")
    shift_time = ShowPrev(p)
    cap = cv2.VideoCapture(p) #動画読み込み 動画の名前
    print(shift_time)
    cap.set(0,shift_time - 2000) #2000ms前からスタート
    video_fps = cap.get(cv2.CAP_PROP_FPS)           # FPS を取得する





    try:
        # フレームを3枚取得してグレースケール変換
        frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

        nextframe = cap.read()[1] #再生用に1枚抜く

        frame3 = cv2.cvtColor(nextframe, cv2.COLOR_RGB2GRAY)
        h, w = frame1.shape
    except:
        _ = 0





    frame_number = 0
    #移動量用のフラッグ設定
    flag_init = 0 #初回フレーム判定のためのフラグ

    plt.subplot(2,1,1)
    plt.plot(0,0,'b',marker='.')##画像の右端の点を白でプロットすることでグラフの描画エリアと画像サイズを合わせる
    plt.plot(w,0,'b',marker='.')
    plt.plot(w,-1*h,'b',marker='.')
    plt.plot(0,-1*h,'b',marker='.')

    plt.plot([0,w],[0,0],'b',linewidth = 0.5) #画像の境界線をプロット
    plt.plot([0,0],[0,-1*h],'b',linewidth = 0.5)
    plt.plot([0,w],[-1*h,-1*h],'b',linewidth = 0.5)
    plt.plot([w,w],[0,-1*h],'b',linewidth = 0.5)




    while(frame_number < video_fps * 4):
        time_stamp = datetime.datetime.now()
        frame_number += 1
        # フレーム間差分を計算
        mask = frame_sub(frame1, frame2, frame3, th=40)
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


        if(sleepcnt >= 10 and sleepcnt < 20): #フリーズ
            #サブプロットに変更
            plt.subplot(2,1,1)
            plt.plot(x,-1*y,'y',marker='.',markersize=7)#黄丸をプロット
            cv2.circle(showframe, (x,y), 15, (0,255,255),-1) #黄丸をカメラ映像に表示
        elif(sleepcnt >= 20): #寝てる
            plt.subplot(2,1,1)
            plt.plot(x,-1*y,'g',marker='.',markersize=7)#緑丸をプロット
            cv2.circle(showframe, (x,y), 15, (0,255,0),-1) #緑丸をカメラ映像に表示



        if area > th_param: #面積が閾値より大きければ、重心の座標を更新
            sleepcnt = 0 #眠ってるカウントをリセット
            flag_init = flag_init + 1
            mu = cv2.moments(mask, False)
            try:
                x,y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
                #移動量計算
                distance = np.sqrt((x-before_x)**2+(y-before_y)**2)
                #before_x, before_yが設定されていない状態（一番初め）
                #if flag_moment == 1:
                 #   distance = 0
                    #フラッグ消す
                  #  flag_moment = 0
                #リストにデータ追加
                if(flag_init >= 0):
                    dist_list.append(distance)
                    ran_list.append(frame_number)
                    plt.subplot(2,1,1)
                    plt.plot(x,-1*y,'r',marker='.',markersize=3) #サンプリング点をプロット
                    plt.subplot(2,1,1)
                    plt.plot([before_x,x],[-1*before_y,-1*y],'k',linewidth = 0.5,alpha = 0.5) #線をプロット
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


                    #print(flag_old,flag_new)
                before_x = x
                before_y = y
                flag_old = flag_new #動いてるかどうかフラグ更新
                flag_new = 0

            except:
                _ = 0

        else :   #面積が閾値より小さければ、前回の座標を表示
            if(flag_init >= 1):
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
                if(sleepcnt >= 10 and sleepcnt < 20): #寝てたら
                    f.write("freez")
                elif(sleepcnt >= 20):
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
    dist_list[0] = 0
    #t.cancel()
    cap.release()
    cv2.destroyAllWindows()
    #サブプロットで移動量を描画　横軸はフレーム数
    plt.subplot(2,1,2)
    plt.plot(ran_list, dist_list)
    plt.show()

    #logger(filename, todaydetail)
    print(dist_list)
    #print(len(dist_list))
    maxxxx = max(dist_list)
    shock_point = 0


    for i,j in enumerate(dist_list):
        #print("{0} {1}".format(i,j))
        #print(type(j))
        if j == maxxxx:
            if len(dist_list)/2 -3 <= i <= len(dist_list)/2+3:
                shock_point = i
            else:
                video_fps = int(video_fps)
                #print(video_fps)
                shock_point = len(dist_list) - video_fps *2

    #ショックのフレームナンバーを取得
    #print(shock_point)

    flo_dist = [float(s) for s in dist_list]

    After_shock_sum = 0
    Before_shock_sum = 0

    for x in range(shock_point):
        #print(flo_dist[x])
        Before_shock_sum += flo_dist[x]

    #print("Check")

    for y in range(shock_point, len(dist_list)):
        #print(flo_dist[y])
        After_shock_sum += flo_dist[y]

    print("{0} {1}".format(Before_shock_sum, After_shock_sum))

    change_rate = (After_shock_sum-Before_shock_sum)/(After_shock_sum+Before_shock_sum)
    print(change_rate)

    logger(filename, todaydetail, Before_shock_sum, After_shock_sum, change_rate)





if __name__ == '__main__':
    #動画取得
    os.makedirs("Subprocess", exist_ok=True)
    pathr = "Subprocess/movie_list.dat"
    files = open(pathr)
    movielist = []
    for line in files:
        newl = line.strip()
        if newl == "":
            continue
        else:
            movielist.append(newl)


    print("length {}".format(len(movielist)))
    print("list {}".format(movielist))

    #バッチ処理
    for name in movielist:
        t=threading.Thread(target = IsSleep)
        t.setDaemon(True)
        t.start()
        main(name)
