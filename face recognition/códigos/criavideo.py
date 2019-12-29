'''
Created on 11 de jul de 2019

@author: michel
'''
import numpy as np
import cv2
from cv2 import cvtColor

cap=cv2.VideoCapture(0)#abre a camera

fourcc = cv2.VideoWriter_fourcc(*'XVID')#codeg
out = cv2.VideoWriter('output_part0.avi',fourcc, 20.0, (640,480))#formato e resolucao padrao
st=0# variavel que conta os frames 
a=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        if st==100:#espera a camera estabilizar para comecar a gravar
            a=1
            print("comecou")
        if a==1:
            cv2.imwrite("data1/img"+str(st)+".jpg",frame)#grava frame  a frame e salva na pasta data1
            out.write(frame)
            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        st+=1 
           
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):#quando a tecla q for pressionada termina o programa
            break
    else:
        break

# fecha as janelas
cap.release()
out.release()
cv2.destroyAllWindows()
