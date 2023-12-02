import cv2 as cv
#img=cv.imread('photo/arijit.jpg')
#cv.imshow('image', img)

#gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('gray', gray)

capture = cv.VideoCapture(0)
while True:
    isTrue,frame=capture.read()
    haar_cascade=cv.CascadeClassifier('faces.xml')
    faces_rect=haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=6)
    print(f"Number of faces found = {len(faces_rect)}")

    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    cv.imshow('video',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

cv.waitKey(0)