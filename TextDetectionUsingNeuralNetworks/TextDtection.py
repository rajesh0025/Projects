import numpy as np
import cv2
import os
from numpy.core.fromnumeric import size
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


path = "TextDetectionUsingNeuralNetworks\myData"
testRatio = 0.2
valRatio =0.2

images = []
classNo = []
myList = os.listdir(path)
print("Total no of Classes detected", len(myList))
noOfClasses = len(myList)

print("Importing Classes.....")
for x in range(0, noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))#read each num folder
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)#each image of respective folder
        curImg = cv2.resize(curImg, (32,32))#resize 180/180 to 32/32 because it is computationally expensive
        images.append(curImg)#all images stored
        classNo.append(x)#all class no are stored
    print(x,end= " ")
print(" ")

#create array
images = np.array(images)
classNo = np.array(classNo)

print(images.shape)#value, shape,shape , no of colors(RGB) =3
print(classNo.shape)

#Splittng Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
#Validation
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)

print(X_train.shape, y_train.shape)

# class in y_train
numOfSamples = []
for x in range(0, noOfClasses):
    numOfSamples.append(len(np.where(y_train==x)[0]))#no of images(which are in numerical 0to9) present in each class in y_train
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("No of images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()

print(X_train[9].shape)
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#3 to 1
    img = cv2.equalizeHist(img)#equal light to the image
    img = img/255 # Normalization (restricting the 0-255 to 0-1)
    return img 

#preprocess all the images in X_train using map func and convert to array

X_train = np.array(list(map(preProcessing, X_train)))
X_train = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))
"""img = X_train[9]
img = cv2.resize(img, (300,300))
cv2.imshow("preprocessed", img)
cv2.waitKey(0)
print(img.shape)"""


#we need to add depth 1 for running cnn properly add 4th parameter as depth
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)


#Augument the data --for looking real
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)#augument it send it back

#encoding
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

def myModel():
    noOfFilters=60
    sizeOfFilter1=(5,5)
    sizeOfFilter2=(3,3)
    sizeOfPool=(2,2)
    noOfNode=500

     


