# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def main():
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
        plt.plot(w,-1*h,marker='.')
    except:
        _ = 0
    
    
   

    #前回の座標値を記憶する変数
    before_x = 0
    before_y = 0

    cnt = 0
    while(cap.isOpened()):
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

        
        if area > 400: #面積が閾値より大きければ、重心の座標を更新
            mu = cv2.moments(mask, False)
            try:
                x,y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
                plt.plot(x,-1*y,'r',marker='.',markersize=3)
                plt.plot([before_x,x],[-1*before_y,-1*y],'g',linewidth = 0.5)
                before_x = x
                before_y = y
                cv2.circle(mask, (x,y), 8, (255,255,255),-1)

            except:
                _ = 0

        else :   #面積が閾値より小さければ、前回の座標を表示
            cv2.circle(mask, (before_x,before_y), 8, (255,255,255),-1)


        # 結果を表示
        #cv2.imshow("Frame2", frame2)
        cv2.imshow("Mask", mask)

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

        cnt = cnt+1
    cap.release()
    cv2.destroyAllWindows()
    plt.show()


if __name__ == '__main__':
    main()
