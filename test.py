# -*- coding: utf-8 -*-
import cv2
import numpy as np

# フレーム差分の計算
def frame_sub(img1, img2, img3, th):
    # フレームの絶対差分
    diff1 = cv2.absdiff(img1, img2)
    diff2 = cv2.absdiff(img2, img3)

    # 2つの差分画像の論理積
    diff = cv2.bitwise_and(diff1, diff2)

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

    while(cap.isOpened()):
        # フレーム間差分を計算
        mask = frame_sub(frame1, frame2, frame3, th=30)

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