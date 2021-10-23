import cv2 


thres = 0.5


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


classNames =[]
classFile = 'obj detection\coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip("\n").split("\n")
configPath = "obj detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "obj detection/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath) #get ids of img
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidance, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0,255,0),thickness=2)
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(confidance), (box[0]+150, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Output", img)
    cv2.waitKey(1)

 