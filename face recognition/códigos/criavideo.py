'''
Created on 11 de jul de 2019

@author: michel
'''
import numpy as np
import cv2
from cv2 import cvtColor

cap=cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_part0.avi',fourcc, 20.0, (640,480))
st=0
a=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        if st==100:
            a=1
            print("comecou")
        if a==1:
            cv2.imwrite("data1/img"+str(st+150)+".jpg",frame)
            out.write(frame)
            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        st+=1 
           
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
