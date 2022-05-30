import cv2
import numpy as np
import glob
import os

# importing module 
import youtube_dl 

ydl_opts = {} 

def dwl_vid(): 
    with youtube_dl.YoutubeDL(ydl_opts) as ydl: 
        ydl.download([zxt]) 


link_of_the_video = input("Copy & paste the URL of the YouTube video you want to download:- ") 
zxt = link_of_the_video.strip() 

dwl_vid() 

list_of_files = glob.glob('/Users/devesh/*') 
x = max(list_of_files, key=os.path.getctime)

face_cascade = cv2.CascadeClassifier('HCI_face.xml')
eye_cascade = cv2.CascadeClassifier('HCI_eye.xml')

cap = cv2.VideoCapture(x)

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

os.remove(x)
cap.release()
cv2.destroyAllWindows()