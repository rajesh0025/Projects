import cv2
import time
import numpy as np
import HandTrackingModule as htm 
import math 
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findhands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0 :
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2] # the x and v values of index 4 ex - lmList[4] = [4,344,211]
        x2, y2 = lmList[8][1], lmList[8][2] # ||
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0,255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 177), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        #Hand range 25 to 200 (min to max)
        #Volume range -65 to 0 (min to max)

        vol = np.interp(length, [50, 150], [minVol, maxVol])
        volBar = np.interp(length, [25, 200], [400, 150])
        volPer = np.interp(length, [25, 200], [0, 100])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        

        if length <= 25:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40,450), cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 0), 3)
    cv2.putText(img, "Gesture Volume Control", (100,30), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0), 2)
    
    cv2.imshow("img", img)
    cv2.waitKey(1)