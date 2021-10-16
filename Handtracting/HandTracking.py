import cv2 
import mediapipe as mp 
import time 

cap = cv2.VideoCapture(0)#wecam no 

mpHands = mp.solutions.hands#default
hands = mpHands.Hands()#each hand
mpDraw = mp.solutions.drawing_utils#drawing points

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#as opencv allows use to BGR convert to rgb
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks: # 2 hands 
        for handlms in results.multi_hand_landmarks: #2 loops
            for id, lm in enumerate(handlms.landmark):# we can get 21 ids (points),x,y,z values in image ratios which are later converted into pixels
                h, w, c = img.shape# height n width and center
                cx, cy =int(lm.x *w), int(lm.y *h)#getting pixels
                if id == 4: # we can access each point
                    cv2.circle(img, (cx, cy), 15, (255,0,255),cv2.FILLED)


            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)#single hand with original img and landmarks (just like points) , draw draws those points 

    cv2.imshow("Image", img)
    cv2.waitKey(1) 