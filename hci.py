import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('HCI_face.xml')
eye_cascade = cv2.CascadeClassifier('HCI_eye.xml')

cap = cv2.VideoCapture(0)

while True :
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces :
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        wid= int( x + w/2 )
        hei= int (y + h/3)
        cv2.rectangle(img, (wid,hei), (wid+5,hei+5), (0,0,0), -1 )
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 20)
        for (ex, ey, ew, eh) in eyes :
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == 27 :
        break

cap.release()
cv2.destroyAllWindows()