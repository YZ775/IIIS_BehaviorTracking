import numpy as np
import cv2
import math

ini_x = 0
ini_y = 0

#マウスのクリックイベントを設定
def mouse_event(event,x,y,flags,param):
    #初期値を作成
    global ini_x
    global ini_y

    if event == cv2.EVENT_LBUTTONDOWN:
        #初期値を設定する
        ini_x = x
        ini_y = y
        print("{} {}".format(ini_x, ini_y))

#####################################


if __name__ == "__main__":
    cap = cv2.VideoCapture("../sample.mov")
    #cap = cv2.VideoCapture("../wire.mov")
    ret, frame = cap.read()


    #####################################
    #初期座標を取得
    cv2.imshow("initialize first center position", frame)
    while True:
        key = cv2.waitKey(1)&0xff
        cv2.setMouseCallback("initialize first center position",mouse_event)
        #print("{} {}".format(before_x, before_y))

        if key == 13:
            break
    cv2.destroyAllWindows()
    #####################################

    #####################################
    #領域を設定
    #center positionからのwide引数とheight引数
    cw = 150
    ch = 150
    ori_y = ini_y - cw
    if ori_y < 0:
        ori_y = 0

    ori_x = ini_x -ch
    if ori_x < 0:
        ori_x = 0

    track_window = (ori_y,ori_x,ch,cw)
    #####################################

    # set up the ROI for tracking
    roi = frame[ori_x:ori_x+cw, ori_y:ori_y+ch]
    before_center = (ori_x + cw / 2, ori_y + ch / 2)

    before_x = ori_x
    before_y = ori_y
    before_w = cw
    before_h = ch

    hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((20.,20.,20.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    frame_count = 0
    kernel = np.ones((20,20),np.uint8)

    while(1):
        ret ,frame = cap.read()
        frame_count += 1

        """
        if frame_count < 600:
            continue
        """

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            #膨張・収縮処理
            opening = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
            area = cv2.countNonZero(opening)
            if area < 6000:
                print("Lost")

            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(opening, track_window, term_crit)
            cv2.imshow("erosion", opening)
            # Draw it on image
            x,y,w,h = track_window

            #外乱を抑制するための準備
            center  = (x + w/2, y+h/2)
            distance = math.sqrt((center[0] - before_center[0])**2 + (center[1] - before_center[1])**2)


            #トラックwindowが激しく動く場合，トラックwinodwを固定
            if distance > 200:
                img2 = cv2.rectangle(frame, (before_x,before_y), (before_x+before_w,before_y+before_h), 255,2)


            #トラックwinodwが激しくない，つまりトラッキングできている状態なら更新作業
            else:
                img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                before_center = center

                before_x = x
                before_y = y
                before_w = w
                before_h = h

            cv2.imshow('img2',img2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



        #オーバーフローの防止
        if frame_count > 100000:
            frame_count = 11


    cv2.destroyAllWindows()
    cap.release()
