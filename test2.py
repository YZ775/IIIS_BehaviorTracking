# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import threading
import sys
import os
import re

#####可能性 matplotlib # WARNING:
#MatplotlibDeprecationWarning:
#Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.
#In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.

sleepcnt = 0  #タイマー割り込み時に動いていない場合をカウントしていく
ContinueFlag = 1 #タイマー割り込みを続けるかどうかのフラグ
th_param = 400 #マスクの閾値


#####################################
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
#####################################



#####################################
# フレーム差分の計算

def frame_sub(img1, img2, img3, th):
    #フレームの正規化
    out1 = cv2.equalizeHist(img1)
    out2 = cv2.equalizeHist(img2)
    out3 = cv2.equalizeHist(img3)



    # フレームの絶対差分
    diff1 = cv2.absdiff(out1, out2)
    diff2 = cv2.absdiff(out2, out3)

    # 2つの差分画像の論理積
    diff = cv2.bitwise_xor(diff1, diff2)

    # 二値化処理
    diff[diff < th] = 0
    diff[diff >= th] = 255

    # メディアンフィルタ処理（ゴマ塩ノイズ除去）
    mask = cv2.medianBlur(diff, 3)

    return  mask
#####################################



#####################################
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
        if key == ord('f'):
            ChangeBar(ofst+100)
            cv2.setTrackbarPos("Time[ms]","prev", ofst)
            #print(ofst)

        if key == ord('a'):
            ChangeBar(ofst-100)
            cv2.setTrackbarPos("Time[ms]","prev", ofst)
            #print(ofst)

        if key == ord('d'):
            ChangeBar(ofst+10)
            cv2.setTrackbarPos("Time[ms]","prev", ofst)
            #print(ofst)

        if key == ord('s'):
            ChangeBar(ofst-10)
            cv2.setTrackbarPos("Time[ms]","prev", ofst)
            #print(ofst)

        if key == 13:
            break

    cap_prev.release()
    cv2.destroyAllWindows()
    return ofst
#####################################



#####################################
#前回の座標値を記憶する変数
before_x = 0
before_y = 0
x = 0
y = 0
flag_old = 0
flag_new = 0
folder_path = ""

#####################################

#出力ファイルにmotion_indexを書き込む
def logger(movie_info, b_s_s, a_s_s, late, plott, framer):
    #ログの処理
    #f_before = open(f_path, 'r')

    global folder_path

    # output先のパスが書かれたファイルを開く
    output_place = os.path.join("Subprocess", "output_place.dat")
    print(output_place)
    #output_place = os.getcwd + "\\Subprocess" + "\\output_place.dat"
    if not os.path.isfile(output_place):
        print("Output folder is not found.\nFiles will be saved in log_after.")
        folder_path = os.path.join("log_after")
    else:
        file_out = open(output_place,"r")
        folder_path = file_out.readline()
        folder_path.replace("\n","")
        file_out.close()
        #os.remove(output_place)
    if folder_path == "":
        print("Output folder is not found.\nFiles will be saved in log_after.")
        folder_path = os.path.join("log_after")


    # os.makedirs("log_after", exist_ok=True)
    #text = os.path.join(folder_path,"after-", day_name, ".txt")
    #動画ファイル名を抽出 -- マウスのデータ

    text = folder_path + "/after-" + movie_info + ".txt"
    print(folder_path)

    f_after  = open(text, 'w')

    """
    5/15
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
    """

    f_after.write("Before_Shock Sum of Motion Index: ")
    f_after.write(str(b_s_s))
    f_after.write("\n")

    f_after.write("After_Shock Sum of Motion Index: ")
    f_after.write(str(a_s_s))
    f_after.write("\n")

    f_after.write("A-B/A+B: ")
    f_after.write(str(late))
    f_after.write("\n")

    f_after.write("#")
    f_after.write("FrameNumber")
    f_after.write(" | ")
    f_after.write("(X,Y)")
    f_after.write(" | ")
    f_after.write("wake/sleep/freez")
    f_after.write("\n")


    count = 0
    for i, framecount in enumerate(framer):
        f_after.write(str(framecount))
        f_after.write(" | ")

        xplot = plott[count]
        yplot = plott[count+1]
        judge = plott[count+2]

        f_after.write(str(xplot))
        f_after.write(" | ")
        f_after.write(str(yplot))
        f_after.write(" | ")
        f_after.write(str(judge))
        f_after.write("\n")

        count = count +3
    f_after.close()
#####################################



#####################################
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
    #座標保存用リスト
    plot_list = []

    #todaydetail = datetime.datetime.today()
    #loggerへの引数
    #todaydetail = str(todaydetail.year) + str(todaydetail.month) + str(todaydetail.day) + str(todaydetail.hour) + str(todaydetail.minute) + str(todaydetail.second)

    # print("Enter Threshold　parametar")
    # th_param = int(input("default:400 \n >>"))
    # print("\n")

    #動画設定とショックポイントを設定
    ##############################################################################
    th_param = 400 # 仮です。
    select = 1
    p = movie_path

    #動画のパスのみを抽出する
    movie_data = re.split('[\/,"\\"]',p)
    movie_data_ex = movie_data[-1]
    exex = movie_data_ex.split("\\")
    comp_path = exex[-1]

    print("\n")

    print("###############################################################")
    print("set shock point by using seekbar and keyboard\n")

    print("--------------------    Operation Keys    -------------------")
    print("|\t\t    Enter : Done\t\t\t\t|")
    print("|A :-100[ms]\tS : -10[ms]\tD : +10[ms]\tF : +100[ms]\t|")
    print("|    <<\t\t    <\t\t    >\t\t    >>\t\t|")
    print("------------------------------------------------------------------")
    shift_time = ShowPrev(p)
    cap = cv2.VideoCapture(p) #動画読み込み 動画の名前
    print(shift_time)
    cap.set(0,shift_time - 2000) #2000ms前からスタート
    video_fps = cap.get(cv2.CAP_PROP_FPS)           # FPS を取得する
    ##############################################################################


    try:
        # フレームを3枚取得してグレースケール変換
        frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

        nextframe = cap.read()[1] #再生用に1枚抜く

        frame3 = cv2.cvtColor(nextframe, cv2.COLOR_RGB2GRAY)
        h, w = frame1.shape

        #cv2.imwrite("sample.jpg", frame2)
    except:
        _ = 0


    frame_number = 0
    #移動量用のフラッグ設定
    flag_init = 0 #初回フレーム判定のためのフラグ


    #マップ描画用
    ##############################################################################
    plt.subplot(2,1,1)
    plt.plot(0,0,'b',marker='.')##画像の右端の点を白でプロットすることでグラフの描画エリアと画像サイズを合わせる
    plt.plot(w,0,'b',marker='.')
    plt.plot(w,-1*h,'b',marker='.')
    plt.plot(0,-1*h,'b',marker='.')

    plt.plot([0,w],[0,0],'b',linewidth = 0.5) #画像の境界線をプロット
    plt.plot([0,0],[0,-1*h],'b',linewidth = 0.5)
    plt.plot([0,w],[-1*h,-1*h],'b',linewidth = 0.5)
    plt.plot([w,w],[0,-1*h],'b',linewidth = 0.5)
    ##############################################################################


    ##############################################################################
    #メイン処理
    while(cap.get(cv2.CAP_PROP_POS_MSEC) < shift_time + 2000):
    #while(frame_number < video_fps * 4):
        time_stamp = datetime.datetime.now()
        frame_number += 1
        # フレーム間差分を計算
        mask = frame_sub(frame1, frame2, frame3, th=40)
        #マスクの白部分をカウントする
        area = cv2.countNonZero(mask)
        showframe = nextframe.copy()
        showframe = cv2.cvtColor(showframe,cv2.COLOR_RGB2GRAY)
        showframe = cv2.equalizeHist(showframe)
        #showfrmae = (showframe - np.mean(showframe))/np.std(showframe)*16+64


        #フリーズとスリープ判定
        ######################################################
        if(sleepcnt >= 10 and sleepcnt < 20):
            #サブプロットに変更
            plt.subplot(2,1,1)
            plt.plot(x,-1*y,'y',marker='.',markersize=7)#黄丸をプロット
            cv2.circle(showframe, (x,y), 15, (0,255,255),-1) #黄丸をカメラ映像に表示

        elif(sleepcnt >= 20): #寝てる
            plt.subplot(2,1,1)
            plt.plot(x,-1*y,'g',marker='.',markersize=7)#緑丸をプロット
            cv2.circle(showframe, (x,y), 15, (0,255,0),-1) #緑丸をカメラ映像に表示
        #######################################################


        #動きがある場合
        ######################################################
        if area > th_param: #面積が閾値より大きければ、重心の座標を更新
            sleepcnt = 0 #眠ってるカウントをリセット
            flag_init = flag_init + 1
            mu = cv2.moments(mask, False)

            try:
                x,y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
                #移動量計算
                distance = np.sqrt((x-before_x)**2+(y-before_y)**2)

                #リストにデータ追加
                if(flag_init >= 0):
                    dist_list.append(distance)
                    ran_list.append(frame_number)
                    plt.subplot(2,1,1)
                    plt.plot(x,-1*y,'r',marker='.',markersize=3) #サンプリング点をプロット
                    plt.subplot(2,1,1)
                    plt.plot([before_x,x],[-1*before_y,-1*y],'k',linewidth = 0.5,alpha = 0.5) #線をプロット
                    cv2.circle(showframe, (x,y), 7, (0,0,255),-1)

                    #f.write(str(frame_number))
                    #f.write(" | ")
                    #if (select == "1"):#ビデオ読み込みの場合，タイムスタンプはnull
                    #    f.write("null")
                    #else :
                    #    f.write(str(time_stamp))
                    #f.write(" | ")
                    plot_list.append(str(x))
                    #f.write(str(x))
                    #f.write(",")
                    plot_list.append(str(y))
                    #f.write(str(y))
                    #f.write(" | ")
                    #f.write("wake")
                    plot_list.append("wake")
                    #f.write("\n")


                    #print(flag_old,flag_new)
                before_x = x
                before_y = y
                flag_old = flag_new #動いてるかどうかフラグ更新
                flag_new = 0

            except:
                _ = 0

        ######################################################


        #動きがない場合
        ######################################################

        else :   #面積が閾値より小さければ、前回の座標を表示
            if(flag_init >= 1):
                cv2.circle(showframe, (before_x,before_y), 7, (255,0,0),-1)
                #f.write(str(frame_number))
                #f.write(" | ")
                #リストにデータ追加
                distance = 0
                dist_list.append(distance)
                ran_list.append(frame_number)

                #if (select == "1"): #ビデオ読み込みの場合，タイムスタンプはnull
                #    f.write("null")
                #else:
                #    f.write(str(time_stamp))
                #f.write(" | ")
                #f.write(str(before_x))
                plot_list.append(before_x)
                #f.write(",")
                #f.write(str(before_y))
                #f.write(" | ")
                plot_list.append(before_y)


                if(sleepcnt >= 10 and sleepcnt < 20): #寝てたら
                    #f.write("freez")
                    plot_list.append("freez")
                elif(sleepcnt >= 20):
                    plot_list.append("sleep")
                    #f.write("sleep")
                else:
                    plot_list.append("wake")
                    #f.write("wake")

                #f.write("\n")

            flag_old = flag_new #動いてるかどうかフラグ更新
            flag_new = 1

        ######################################################



        # 結果を表示
        cv2.imshow("showframe", showframe)

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
        cv2.waitKey(10)

    #f.close()
    ContinueFlag = 0
    calcurate_flag = 0
    try:
        dist_list[0] = 0
    except:
        calcurate_flag = 1
        print("No shock point")
    #t.cancel()
    cap.release()
    cv2.destroyAllWindows()

    #サブプロットで移動量を描画　横軸はフレーム数
    plt.subplot(2,1,2)
    plt.plot(ran_list, dist_list)

    ##############################################################################
    #メイン処理を終了



    #MotionIndexを計算
    ######################################################
    if calcurate_flag == 0:
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

        try:
            change_rate = (After_shock_sum-Before_shock_sum)/(After_shock_sum+Before_shock_sum)
        except:
            change_rate = 0

        print(change_rate)
        #########################################################

        #ログに書き込み作業
        logger(comp_path, Before_shock_sum, After_shock_sum, change_rate, plot_list, ran_list)
        #outputfolderに画像を出力
        picturepath = folder_path + "/" + comp_path + ".png"
        #print(picturepath)
        plt.savefig(picturepath)
        plt.show()
    else:
        print("No Calucurate Data")






##############################################################################
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


    #print("length {}".format(len(movielist)))
    #print("list {}".format(movielist))

    #バッチ処理
    for name in movielist:
        t=threading.Thread(target = IsSleep)
        t.setDaemon(True)
        t.start()
        main(name)
##############################################################################
