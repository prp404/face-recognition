import cv2 as cv
import numpy as np

def rescaleFrame(frame,scale=0.3):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimension=(width,height)
    return cv.resize(frame,dimension,interpolation=cv.INTER_AREA)

haar_cascade=cv.CascadeClassifier('faces.xml')
people = ['Parth Parlikar', 'Pradnya Parlikar','Rajiv Parlikar']
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img=cv.imread(r'photo/arijit.jpg')
img=rescaleFrame(img)
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Family', gray)

faces_rect=haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6)
for (x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label={people[label]} with a confidence of {confidence}')
    cv.putText(img, str(people[label]),(20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img,(x,y), (x+w, y+h), (0,255,0), thickness=2)
cv.imshow('Detected faces', img)
cv.waitKey(0)