import cv2
import time
 
cap = cv2.VideoCapture(1)
 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_FPS, 30)           # カメラFPSを60FPSに設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # カメラ画像の横幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # カメラ画像の縦幅を720に設定
while True:
 
    ret,frame = cap.read()
 
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    cv2.imshow('frame',frame)

    start = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    process_time = time.time() - start
    print(process_time)
 
cap.release()
cv2.destroyAllWindows()
