import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import threading
import re
import sys
import random
import os

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

########################################################################

# フレーム差分の計算 ここにフレームをぶち込む
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

####################################################

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

#################################################

#前回の座標値を記憶する変数
before_x = 0
before_y = 0
x = 0
y = 0
flag_old = 0
flag_new = 0

########################################################
#ログの後処理

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
###########################################################

def movie_sub():

    #動画のパスをすべて取得
    path_list = os.listdir(".\movie")
    print(path_list)

    #動画のショックポイントをリストにて取得
    shock_time = []

    #パラメータ入力　必要なときコメント解除
    #print("Enter Threshold　parametar")
    #th_param = int(input("default:400 \n >>"))
    #print("\n")

    select = 1
    #p = input("enter video path\n>> ")

    print("########Operation Key############")
    print("w: +10[ms] a:-100[ms] s:-10[ms] d:+100[ms]")

    #####################

    #全動画のショックのポイントを取得する
    for i in range(0,4):

        path = "./movie/" + str(path_list[i])
        print(path)

        shift_time = ShowPrev(path)

        #必要な場合はコメント解除
        #cap = cv2.VideoCapture(path)
        print(shift_time)
        #frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) #総フレーム数を取得
        #video_fps = cap.get(cv2.CAP_PROP_FPS)
        #frame_count = int(frame_count)
        #print("{0}".format(frame_count))

        #W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        #H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #fpss = cap.get(cv2.CAP_PROP_FPS)
        #cap.set(0,shift_time - 2000) #2000ms前からスタート

        shock_time.append(shift_time)

    ####################


    #動画をフレームで切り出して標準化する
    movies_mean = 0
    movies_std = 0

    means = []
    stds = []

    for it in range(0,4):
        #動画のパスを取得
        path = "./movie/" + str(path_list[it])
        print(path)

        cap = cv2.VideoCapture(path)

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) #総フレーム数を取得
        frame_count = int(frame_count)
        #フレーム数を表示
        print("frame {0}".format(frame_count))

        W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fpss = cap.get(cv2.CAP_PROP_FPS)

        #サンプリング20
        rand_list = random.sample(range(1,frame_count), k = 10)
        #サンプルのリストを表示
        print(rand_list)

        count = 0

        #1本の動画のサンプリングフレームの平均と分散を取得
        movie_mean = []
        movie_std = []

        while count <= frame_count:
            ret, frame = cap.read()
            count += 1

            if count in rand_list:
                if ret == True:

                    movie_mean.append(np.mean(frame)) #フレームごとの平均
                    movie_std.append(np.std(frame)) #フレームごとの分散

        #一時的に開放
        cap.release()
        cv2.destroyAllWindows()

        leng = len(movie_mean)
        #print(leng)

        means.append(int(sum(movie_mean)/leng)) #動画ごとの平均
        stds.append(int(sum(movie_std)/leng)) #動画ごとの分散

    #全動画の平均と分散
    len_mov  = len(means)
    movies_mean = int(sum(means)/len_mov)
    movies_std = int(sum(stds)/len_mov)
    print("All movie mean std")
    #print("{0} {1}".format(movies_mean, movies_std))
    cap.release()
    cv2.destroyAllWindows()

    return movies_mean, movies_std, shock_time

##################################################################


def main():

    global before_x #グローバル変数であることを宣言
    global before_y
    global x
    global y
    global flag_old
    global flag_new
    global sleepcnt
    global ContinueFlag
    #移動量保存用リスト 初期化はしないでよい？
    dist_list = []
    ran_list = []

    #動画のパスをすべて取得
    path_list = os.listdir(".\movie")
    print(path_list)

    result = movie_sub()
    print(result)
    movies_mean = result[0]
    movies_std = result[1]
    shock_time = result[2]
    print("test3 mean {0} std {1} time {2}".format(movies_mean, movies_std, shock_time))

#########################################################
    #動画のログファイル作成　ログファイルは1つのみとする
    todaydetail = datetime.datetime.today()
    todaydetail = str(todaydetail.year) + str(todaydetail.month) + str(todaydetail.day) + str(todaydetail.hour) + str(todaydetail.minute) + str(todaydetail.second)

    os.makedirs("log", exist_ok=True)
    filename= "log/Movie" + todaydetail + ".txt"
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

#####################################################

    #平均・分散を取得 -- フレームの操作に処理をシフト
    ####################################

    #フレームに行うメインの処理

    after_list = []
    before_list = []
    change_list = []

    for ix in range(0,4):
        paths = "./movie/" + str(path_list[ix])
        print(paths)

        caps = cv2.VideoCapture(paths)
        assert caps.isOpened(), 'can not open'

        shock_p = int(shock_time[ix])
        print(shock_p)

        video_fps = caps.get(cv2.CAP_PROP_FPS)
        #例外処理を後々
        caps.set(0, shock_p - 2000)

        #Contiue Point
        ret, frame = caps.read()
        while (ret and frame_number < video_fps * 4):
            #frame = (frame - movies_mean)/movies_std #int(16) + int(64)
            frame = (frame - np.mean(frame))/np.std(frame)*int(30) + int(100)
            #print(frame.dtype)
            frames = frame.astype(np.float32)


            h, w, t = frames.shape
            cv2.imshow("frame", frames)

            # フレームを3枚取得してグレースケール変換
            frame1 = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

            #nextframe = caps.read()[1] #再生用に1枚抜く
            nextframe = frames

            frame3 = cv2.cvtColor(nextframe, cv2.COLOR_BGR2GRAY)
            #h, w = frame1.shape

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


                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                ret,frame = caps.read()




            ContinueFlag = 0
            #1次的に解法
        caps.release()
        cv2.destroyAllWindows()
            #サブプロットで移動量を描画　横軸はフレーム数
        plt.subplot(2,1,2)
        plt.plot(ran_list, dist_list)
        plt.show()

        #logger(filename, todaydetail)

        dist_list[0] = 0
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
                    #全動画のvideo fps が取れているのかを確認する必要がある / あるいは、ここでとってしまうか。
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

    #各動画のafter / before / change を書込み
        after_list.append(After_shock_sum)
        before_list.append(Before_shock_sum)
        change_list.append(change_rate)

    f.write("\n")
    f.write("All Movie LOG")
    f.write("movie name")
    f.write("  |  ")
    f.write("Before Shock")
    f.write("  |  ")
    f.write("After Shock")
    f.write("  |  ")
    f.write("Rate")
    f.write("\n")

    for iz in range(0,4):
        f.write(path_list[iz])
        f.write("  ")
        f.write(before_list[iz])
        f.write("  ")
        f.write(after_list[iz])
        f.write("  ")
        f.write(change_list[iz])
        f.write("\n")

    f.close()

    #logger(filename, todaydetail, Before_shock_sum, After_shock_sum, change_rate)

if __name__ == '__main__':
    t=threading.Thread(target = IsSleep)
    t.setDaemon(True)
    t.start()
    main()

##########################################################################
