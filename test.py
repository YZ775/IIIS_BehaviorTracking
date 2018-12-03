# -*- coding: utf-8 -*-
import cv2
import numpy as np

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
    # カメラのキャプチャ
    cap = cv2.VideoCapture(0)

    # フレームを3枚取得してグレースケール変換
    frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

    #前回の座標値を記憶する変数
    before_x = 0
    before_y = 0

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

        #面積が閾値より大きければ、重心の座標を更新
        if area > 600:
            mu = cv2.moments(mask, False)
            try:
                x,y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
                before_x = x
                before_y = y
                cv2.circle(mask, (x,y), 20, (255,255,255),-1)
            except:
                print(" ")

        #面積が閾値より小さければ、前回の座標を表示
        else :
            cv2.circle(mask, (before_x,before_y), 20, (255,255,255),-1)


        # 結果を表示
        cv2.imshow("Frame2", frame2)
        cv2.imshow("Mask", mask)

        # 3枚のフレームを更新
        frame1 = frame2
        frame2 = frame3
        frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

        # qキーが押されたら途中終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
