'''
Created on 11 de jul de 2019

@author: michel
'''
import numpy as np
import cv2

cap = cv2.VideoCapture('henrique/henrique_novo.mp4')#abre o video "henrique_novo.mp4" dentro da pasta henrique.
imcout=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        cv2.imwrite("henrique/img"+str(imcout)+".jpg",frame)#escreve na pasta "henrique" as imagens de "img0.jpg" até "imgn.jpg"
        imcout+=1
    else:
        break

cap.release()
print("terminou!!!!, imagens salvas!")
