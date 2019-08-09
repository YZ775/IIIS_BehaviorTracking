# -*- coding: utf-8 -*-
import cv2
import numpy as np
import serial
from serial.tools import list_ports
import time

def select_port():
    ser = serial.Serial()
    ser.baudrate = 9600    # ArduinoのSerial.beginで指定した値
    ser.timeout = 0.1       # タイムアウトの時間

    ports = list_ports.comports()    # ポートデータを取得

    devices = [info.device for info in ports]

    if len(devices) == 0:
        # シリアル通信できるデバイスが見つからなかった場合
        print("error: device not found")
        return None
    elif len(devices) == 1:
        print("only found %s" % devices[0])
        ser.port = devices[0]
    else:
        # ポートが複数見つかった場合それらを表示し選択させる
        for i in range(len(devices)):
            print("%3d: open %s" % (i,devices[i]))
        print("input number of target port >> ",end="")
        num = int(input())
        ser.port = devices[num]

    # 開いてみる
    try:
        ser.open()
        return ser
    except:
        print("error when opening serial")
        return None


class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)

    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):

        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent

    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]

    #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]

    #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])


def CreateMask(img,centor_hsv,centor_hsv2):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    #hsv_min = np.array([0,0,0])
    #hsv_max = np.array([330,45,29])
    hsv_min = centor_hsv - np.array([20,50,30])
    hsv_max = centor_hsv + np.array([20,120,30])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)


    # 赤色のHSVの値域2
    hsv_min2 = centor_hsv2 - np.array([20,50,30])
    hsv_max2 = centor_hsv2 + np.array([20,120,30])
    mask2 = cv2.inRange(hsv, hsv_min2, hsv_max2)

    kernel = np.ones((5,5),np.uint8)
    return cv2.medianBlur(cv2.dilate(cv2.erode(mask1+mask2,kernel,iterations = 1),kernel,iterations = 1),15)
    #return mask1


def main():
    print("select Source")
    select = input("0 or 1 : from camera \nelse: video PATH\n>>")
    print("\n")
    if(select == "0" or select == "1"):
        select = int(select)
    #####変数初期化#####
    x = 0
    y = 0
    before_x = 0
    before_y = 0
    distance = np.zeros(10)

    centor_hsv = np.array([0,0,0])
    centor_hsv2 = np.array([0,0,0])

    cap = cv2.VideoCapture(select)
    ret, frame = cap.read()

    mask_1 = CreateMask(frame,centor_hsv,centor_hsv2)
    mask_2 = mask_1
    mask_3 = mask_1

    CAP_W = frame.shape[1]
    CAP_H = frame.shape[0]
    #print(CAP_W)
    #print(CAP_H )

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (2*CAP_W,CAP_H))

    ########serialのopen#######
    print("Select Device")
    ser = select_port()
    print(ser.name)
    #マウスクリックで対象を選択

    MOTION_COF = float(input("Enter motion index coefficient(default:1.0) >>"))

    cv2.namedWindow('Frame')
    frame_temp = frame.copy()
    cv2.imshow("Frame", frame_temp)
    mouseData = mouseParam("Frame")

    flag1 = 0
    flag2 = 0
    while(1):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            if(flag1 ==0):
                #frame_temp = frame.copy()
                print(mouseData.getPos())
                hsv = cv2.cvtColor(frame_temp,cv2.COLOR_BGR2HSV)
                centor_hsv = hsv[mouseData.getY(),mouseData.getX(),:]
                cv2.circle(frame_temp, (mouseData.getX(),mouseData.getY()), 7, (255,0,0),-1)
                cv2.imshow("Frame", frame_temp)
                print(centor_hsv)
                flag1 = 1

        if mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            if(flag2 ==0):
                #frame_temp = frame.copy()
                print(mouseData.getPos())
                hsv = cv2.cvtColor(frame_temp,cv2.COLOR_BGR2HSV)
                centor_hsv2 = hsv[mouseData.getY(),mouseData.getX(),:]
                cv2.circle(frame_temp, (mouseData.getX(),mouseData.getY()), 7, (255,255,0),-1)
                cv2.imshow("Frame", frame_temp)
                print(centor_hsv2)
                flag2 = 1

    cv2.destroyAllWindows()

    flag = 1
    while(cap.isOpened()):

        ret, frame = cap.read()

        mask_3 = CreateMask(frame,centor_hsv,centor_hsv2)

        area = cv2.countNonZero(mask_3)

        #鹿間 追加事項
        #前回マスクと今回のマスクの引き算をして．面積を計算 -> 閾値を超えれば重心等の計算を行う
        #少しだけ隠れていても電圧を出力可能になる
        #寝ている時は，30%を下回る
        #少し隠れていても動いているときは，30%をほとんど下回らない

        #よって，距離が0~20?で，かつperが30%以下なら，寝ている判定をして，電圧0vがベストだと思う．
        if flag == 1:
            before_mask = mask_3
            flag = 0
        if flag == 0:
            diff = cv2.absdiff(mask_3, before_mask)
            motion_area = cv2.countNonZero(diff)
            try:
                per = motion_area / area * 100
            except:
                print("error")

            print("motion_area {}".format(per))


        #鹿間追加事項
        #マスク画像で，最大面積の部分を取得
        #最大面積部分を少し拡大させた局所領域で，画像処理を行い精度向上と高速化を図れるはず
        #とりあえず，コメントアウトした．
        """
        images,contours, hierarchy = cv2.findContours(mask_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area = 0
        best_label = 0
        for i in range(0, len(contours)):
            if len(contours[i]) > 0:
                new_area = cv2.contourArea(contours[i])
                if best_area < new_area:
                    best_area = new_area
                    best_label = i

        #最大面積部分のROI(局所領域)を取得 値は取得できたけど，x軸なのか，y軸なのかが分からない
        print(contours[best_label])
        print("\n")
        prepro = contours[best_label].max(axis=1)
        max_xy = prepro.max(axis=1)
        min_xy = prepro.min(axis=1)

        real_max = max_xy.max()
        real_min = min_xy.min()
        print("max {} min {}".format(real_max, real_min))
        print("best_label:{}, area:{}".format(best_label, best_area))
        """



        before_x = x
        before_y = y
        if(area > 5000):
            mu = cv2.moments(mask_3, False)
            try:
                x,y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
            except:
                _ = 0
        distance = np.roll(distance, 1) #要素数10の配列をシフトしてsumを取ることで矩形波とのconvolution
        distance[0] = MOTION_COF*300000*(np.sqrt((x-before_x)**2+(y-before_y)**2) / (CAP_W*CAP_H))#ピクセル数によって正規化

        distance_str = str(distance.sum())
        print(distance_str)
        ##serial送信####
        ser.write(distance_str.encode('utf-8'))
        ser.write(b'\n')
        ser.reset_output_buffer()

        show_frame = cv2.cvtColor(mask_3,cv2.COLOR_GRAY2RGB)
        cv2.circle(show_frame, (x,y), 7, (0,0,255),-1)

        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (0,0), (400,50), (0,0,0), thickness=-1, lineType=cv2.LINE_8, shift=0)
        cv2.putText(frame_copy, distance_str, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

        out_frame = cv2.hconcat([show_frame,frame_copy])
        cv2.imshow("Mask",out_frame)

        out.write(out_frame)

        before_mask = mask_3



        # qキーが押されたら途中終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    out.release()


if __name__ == '__main__':
    main()
