import cv2 
import mediapipe as mp 
import time 
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)#wecam no 
detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findhands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4])#landmark

    cv2.imshow("Image", img)  
    cv2.waitKey(1) 
 