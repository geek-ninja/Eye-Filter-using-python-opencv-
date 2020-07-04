import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('face_trainer.xml')
eye_cascade = cv2.CascadeClassifier('eye_detect_trainer.xml')
eye_filter = cv2.imread('fire2.png') #You can change the effect here by replacing with your own png image.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            eye_resize = cv2.resize(eye_filter,(ew,eh))
            eyeGray = cv2.cvtColor(eye_resize,cv2.COLOR_BGR2GRAY)
            _,eyeMask = cv2.threshold(eyeGray,135,255,cv2.THRESH_BINARY) #You have to change the threshold value to get the proper visibility of the effect.
            eyeMaskInv = cv2.bitwise_not(eyeMask)
            roi = roi_color[ey:ey+eh,ex:ex+ew]
            eye_bg = cv2.bitwise_and(roi,roi,mask=eyeMaskInv)
            eye_fg = cv2.bitwise_and(eye_resize,eye_resize,mask = eyeMask)
            dst = cv2.add(eye_bg,eye_fg)
            roi_color[ey:ey+eh,ex:ex+ew] = dst
            # cv2.ellipse(roi_color,((2*ex+ew)//2,(2*ey+eh)//2),(ew-(ew//3),eh//2),0,0,360,(255,255,255),-1)
            # cv2.circle(roi_color,((2*ex+ew)//2,(2*ey+eh)//2),8,(0,0,0),-1)
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
