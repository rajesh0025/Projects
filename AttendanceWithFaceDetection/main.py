import cv2
import numpy as np
import face_recognition

imgrayn = face_recognition.load_image_file('AttendanceWithFaceDetection/j1.jpg')
imgrayn = cv2.cvtColor(imgrayn,cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file('AttendanceWithFaceDetection/j2.jpg')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgrayn)[0]
print(faceloc)
encoderayn = face_recognition.face_encodings(imgrayn)[0]
cv2.rectangle(imgrayn, (faceloc[3],faceloc[0],faceloc[1],faceloc[2]),(255,0,255),2)


facelocTest = face_recognition.face_locations(imgtest)[0]
encodeTest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest, (facelocTest[3],facelocTest[0],facelocTest[1],facelocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encoderayn], encodeTest)
faceDis = face_recognition.face_distance([encoderayn], encodeTest)
print(results)
print(faceDis)
cv2.putText(imgtest, f"{results} {round(faceDis[0],2)} ", (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 2)


cv2.imshow('Rayn', imgrayn)
cv2.imshow('test', imgtest)
cv2.waitKey(0)