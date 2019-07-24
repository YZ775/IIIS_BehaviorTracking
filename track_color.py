# -*- coding: utf-8 -*-
import cv2
import numpy as np
import serial

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
        

def CreateMask(img,centor_hsv):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    #hsv_min = np.array([0,0,0])
    #hsv_max = np.array([330,45,29])
    hsv_min = centor_hsv - np.array([20,50,30])
    hsv_max = centor_hsv + np.array([20,120,30])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    """
    # 赤色のHSVの値域2
    hsv_min = np.array([150,127,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)
    """
    kernel = np.ones((5,5),np.uint8)
    return cv2.medianBlur(cv2.dilate(mask1,kernel,iterations = 1),15)
    #return mask1


def main():
    #####変数初期化#####
    x = 0
    y = 0
    before_x = 0
    before_y = 0
    distance = np.zeros(10)

    centor_hsv = np.array([0,0,0])
    cap = cv2.VideoCapture("wire.mov")
    ret, frame = cap.read()
    
    mask_1 = CreateMask(frame,centor_hsv)
    mask_2 = mask_1
    mask_3 = mask_1
    ###############
    ser = serial.Serial('/dev/cu.usbmodem143203', 9600,parity=serial.PARITY_NONE)
    print(ser.name)
    #マウスクリックで対象を選択
    
    frame_temp = frame.copy()
    cv2.imshow("Frame", frame_temp)
    mouseData = mouseParam("Frame")

    while(1):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            frame_temp = frame.copy()
            print(mouseData.getPos())
            hsv = cv2.cvtColor(frame_temp,cv2.COLOR_BGR2HSV)
            centor_hsv = hsv[mouseData.getY(),mouseData.getX(),:]
            cv2.circle(frame_temp, (mouseData.getX(),mouseData.getY()), 7, (255,0,0),-1)
            cv2.imshow("Frame", frame_temp)
            print(hsv[mouseData.getY(),mouseData.getX(),:])
            
        elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            break
    
    cv2.destroyAllWindows()

    while(cap.isOpened()):

        ret, frame = cap.read()

        mask_1 = mask_2
        mask_2 = mask_3
        mask_3 = CreateMask(frame,centor_hsv)
        """
        # フレームの絶対差分
        diff1 = cv2.absdiff(mask_1, mask_2)
        diff2 = cv2.absdiff(mask_2, mask_3)

        # 2つの差分画像の論理積
        diff = cv2.bitwise_xor(diff1, diff2)
        """
       

        area = cv2.countNonZero(mask_3)

        
        before_x = x
        before_y = y
        if(area > 5000):
            mu = cv2.moments(mask_3, False)
            try:
                x,y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
            except:
                _ = 0
        distance = np.roll(distance, 1) #要素数10の配列をシフトしてsumを取ることで矩形波とのconvolution
        distance[0] = np.sqrt((x-before_x)**2+(y-before_y)**2)
        print(distance.sum())
        ser.write(str(distance.sum())) 
        ser.write(b'\n')
        ser.reset_output_buffer()

        show_frame = cv2.cvtColor(mask_3,cv2.COLOR_GRAY2RGB)
        cv2.circle(show_frame, (x,y), 7, (0,0,255),-1)
 
        cv2.imshow("Mask", cv2.hconcat([show_frame, frame]))
        # qキーが押されたら途中終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
