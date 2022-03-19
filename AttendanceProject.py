from base64 import encode
import cv2
from matplotlib import lines
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'ImagesAttendance'
images = [] # names of images
classNames = [] # names of people
myList = os.listdir(path)
#print(myList)
for cl in myList:
    curImg = cv2.imread(path + str("/")+str(cl))
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    #print(classNames)

 
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        lines = f.readlines()
        nameList = []

        for line in lines:
            entry = line.split(',')
            nameList.append(entry[0])
        
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime("%m/%d/%Y, %H:%M:%S")
            sentence_to_be_added = str(name) + str(", ") + str(dateString)
            f.writelines( '\n' + sentence_to_be_added )
           
 
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
encodeListKnown = findEncodings(images)
print("Endoded List Lenght: ", len(encodeListKnown))

# - - - 
cap = cv2.VideoCapture(0) # 0 is the id of the camera which is web cam of the computer
 
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace) # compare faces
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) # compare faces distance
        print(faceDis)
        matchIndex = np.argmin(faceDis) # index of the minimum distance
    
        if matches[matchIndex]:
            name = classNames[matchIndex].upper() # name of the person
            print(name)

            # Face detection and bounding box here. (until line 60)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
            markAttendance(name) # call the function to mark person name to the csv file

    cv2.imshow("Frame",img)
    cv2.waitKey(1)