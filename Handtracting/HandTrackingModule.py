import cv2 
import mediapipe as mp 
import time 


class  handDetector():
    def __init__(self, mode= False, maxHands=2, detectionCon=0.5, trackCon=0.5):#Hand() function params
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#as opencv allows use to BGR convert to rgb
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks: # 2 hands 
            for handlms in self.results.multi_hand_landmarks: #2 loops
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)#single hand with original img and landmarks (just like points) , draw draws those points 
            
        return img 
    
    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks: # 2 hands 
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):# we can get 21 ids (points),x,y,z values in image ratios which are later converted into pixels
                h, w, c = img.shape# height n width and center
                cx, cy =int(lm.x *w), int(lm.y *h)#getting pixels
                lmlist.append([id, cx, cy])
                """if draw: # we can access each point
                    cv2.circle(img, (cx, cy), 10, (25,55,22),cv2.FILLED)"""
        return lmlist  

def main():
    cap = cv2.VideoCapture(0)#wecam no 
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])#landmark

        cv2.imshow("Image", img)  
        cv2.waitKey(1) 
 
   
if __name__ == "__main__":
    main()