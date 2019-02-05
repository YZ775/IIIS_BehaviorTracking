import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import os


path_list = os.listdir(".\movie")
print(path_list)

#動画の平均と分散をリスト
movie_mean = []
movie_std = []

#全動画に処理をおこなう
for i in range(0,4):
    path = "./movie/" + str(path_list[i])
    print(path)


    cap = cv2.VideoCapture(path)

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) #総フレーム数を取得
    frame_count = int(frame_count)
    print("{0}".format(frame_count))

    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fpss = cap.get(cv2.CAP_PROP_FPS)

    #サンプリング20
    rand_list = random.sample(range(1,frame_count), k = 20)
    print(rand_list)

    count = 0

    #1本の動画のサンプリングフレームの平均と分散を取得
    while count <= frame_count:
        ret, frame = cap.read()
        count += 1

        if count in rand_list:
            if ret == True:

                print(np.mean(frame))
                print(np.std(frame))

                movie_mean.append(np.mean(frame)) #平均
                movie_std.append(np.std(frame)) #分散

    #一時的に開放
    cap.release()

    means = 0
    stds = 0

    leng = len(movie_mean)
    print(leng)
    for z in range(leng):

        means += movie_mean[z]
        stds += movie_std[z]

    #動画の平均と分散を取得
    means = int(means/len(movie_mean))
    stds = int(stds/len(movie_std))

    #動画を再合成する
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(str(i) + '.avi', fourcc, fpss, (int(W),int(H)))

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            #フレームの正規化
            frame = (frame - means)/stds*16 + 64

            #書込み
            output.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
