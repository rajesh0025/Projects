import cv2
import numpy as np
import face_recognition
import os 
from datetime import datetime

from numpy.lib.shape_base import _make_along_axis_idx

path = "AttendanceWithFaceDetection\imageAttendance"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")#each path 
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])#get their names 
print(classNames)

#encode original image 
def findEncodings(images):#encode all images
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete")   

#take attendance
def markAttendance(name):
    with open("AttendanceWithFaceDetection\Attendance.csv","r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f"\n{name},{dtString}")


#compare the original image with person in the web cam

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)#resize it to reduce time
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurFrame)#encode

    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurFrame):#iterate through all faces in the image
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)# compare all the imageattandace to the persons in the webcam
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)#find the distance (which has low is the best)
        print(faceDis)#the match image has low dis 
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), 2)
            """cv2.rectangle(img, (x1, y2-35, (x2,y2), (0,255,0),cv2.FILLED))"""
            cv2.putText(img, name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            markAttendance(name)#if matches and not in list ...



    cv2.imshow("Webcam", img)
    cv2.waitKey(1)