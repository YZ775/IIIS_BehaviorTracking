# -*- coding: utf-8 -*-

import cv2

class Tracking_Module:

    def change_video_offset(self,video_obj,offset): #ビデオのオフセットを変えるメソッド
        video_obj.set(cv2.CAP_PROP_POS_MSEC,offset)


    def show_video_prev(self,video_obj): #シークバー付きプレビューを表示するメソッド 戻り値:シークバーの位置
        self.ofst = 0
        def change_bar(val): #シークバーが動いた時に呼ばれる関数
            self.ofst = val
            Tracking_Module.change_video_offset(self,video_obj,val)
            frame = video_obj.read()[1]
            cv2.imshow("Preview",frame) #フレームを更新

        video_frame = video_obj.get(cv2.CAP_PROP_FRAME_COUNT) # フレーム数を取得する
        video_fps = video_obj.get(cv2.CAP_PROP_FPS)           # FPS を取得する
        video_len_sec = int((video_frame / video_fps)*1000) #長さ[ms]を取得
        
        cv2.namedWindow('Preview') #ウインドウを生成
        cv2.createTrackbar("Time[ms]", "Preview", 0, video_len_sec, change_bar) #トラックバーを生成
        frame = video_obj.read()[1]
        cv2.imshow("Preview",frame)

        while True: #キー入力待ち
            key = cv2.waitKey(1)&0xff
            if key == ord('q'):
                break
            elif key == ord('a'):
                cv2.setTrackbarPos("Time[ms]","Preview", self.ofst-100) 

            elif key == ord('d'):
                cv2.setTrackbarPos("Time[ms]","Preview", self.ofst+100) 


        return self.ofst

    def frame_diff(self,frame1,frame2,frame3,th): #フレーム差分を作るメソッド　戻り値は面積,座標
       
        #def frame_sub(img1, img2, img3, th): # フレーム差分の計算
        # フレームの絶対差分
        diff1 = cv2.absdiff(frame1, frame2)
        diff2 = cv2.absdiff(frame2, frame3)

        # 2つの差分画像の論理積
        diff = cv2.bitwise_xor(diff1, diff2)

        # 二値化処理
        diff[diff < th] = 0
        diff[diff >= th] = 255

        # メディアンフィルタ処理（ゴマ塩ノイズ除去）
        mask = cv2.medianBlur(diff, 3)

        area = cv2.countNonZero(mask)
        mu = cv2.moments(mask, False)
        x,y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
        #x = 0
        #y = 0

        return mask,area,x,y