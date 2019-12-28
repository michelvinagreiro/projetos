import numpy as np
import cv2
from cv2 import cvtColor
import os, sys
import time
import pickle
import face_recognition
from keras.preprocessing.image import image 
import tensorflow as tf
from keras.models import load_model
import RPi.GPIO as gpio

cap=cv2.VideoCapture(0)
data=pickle.loads(open("enc_TCC.pickle","rb").read())
st=0
a=0
henr_model=load_model('henrique_saved.h5')

iran_model=load_model('iranilson_saved.h5')

wands_model=load_model('wands_saved.h5')

print("1...,")
henr=cv2.imread("henrique.jpg",3)
boxhenr=face_recognition.face_locations(henr,model="hog")
henr_encod=face_recognition.face_encodings(henr,boxhenr)
print("2...,")
iran=cv2.imread("iranilson.jpg",3)
boxiran=face_recognition.face_locations(iran,model="hog")
iran_encod=face_recognition.face_encodings(iran,boxiran)
print("3...")
wands=cv2.imread("wands.jpg",3)
boxwands=face_recognition.face_locations(wands,model="hog")
wands_encod=face_recognition.face_encodings(wands,boxwands)
gpio.setmode(gpio.BOARD)
gpio.setup(12, gpio.OUT)
gpio.output(12, gpio.LOW)

print("VALENDO!!!!!!!!!!!!!!!!!")
while(cap.isOpened()):
    ret, frame = cap.read()
    frameout=frame.copy()
    boxes=face_recognition.face_locations(frame,model="hog")
    encodings=face_recognition.face_encodings(frame,boxes)
    names=[]
    
    for encoding in encodings:
        matches=face_recognition.compare_faces(data["encodings"],encoding)
        name="Unknow"
        if True in matches:
            matchedidxs=[i for (i, b) in enumerate(matches) if b]
            counts= {}
            for i in matchedidxs:
                name=data["names"][i]
                counts[name]=counts.get(name,0)+1
            name=max(counts, key=counts.get)
        names.append(name)
    for ((top, right, bottom, left),name) in zip(boxes,names):
        if name=="intruso":
            cv2.rectangle(frame, (left, top), (right,bottom), (0,0,255),3)
            y= top - 15 if top - 15>15 else top+15
            
            cv2.putText(frame, name, (left,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
            cv2.putText(frame, 'Pressione "s" para acessar via senha', (100,460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
    
        if name=="henrique":
            dist=face_recognition.face_distance(encoding,henr_encod)
            crp= frame[int(boxes[0][0]):int(boxes[0][2]),int(boxes[0][3]):int(boxes[0][1])]
            crp= cv2.resize(crp,(224,224),interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite("out0.jpg",crp)
            img=image.load_img("out0.jpg", target_size=(224,224))
            img=np.expand_dims(img, axis=0)
            result=henr_model.predict_proba(img)
            if result[0][0]==1.0 and dist<0.45:
                cv2.rectangle(frame, (left, top), (right,bottom), (0,255,0),3)
                y= top - 15 if top - 15>15 else top+15
                cv2.putText(frame, name, (left,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2) 
                gpio.output(12, gpio.HIGH)
                time.sleep(5)
                gpio.output(12, gpio.LOW)
            else:
                cv2.rectangle(frame, (left, top), (right,bottom), (0,0,255),3)
                y= top - 15 if top - 15>15 else top+15
                
                cv2.putText(frame, 'intruso', (left,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                cv2.putText(frame, 'Pressione "s" para acessar via senha', (100,460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
 
        if name=="iranilson":
            dist=face_recognition.face_distance(encoding,iran_encod)
            crp= frame[int(boxes[0][0]):int(boxes[0][2]),int(boxes[0][3]):int(boxes[0][1])]
            crp= cv2.resize(crp,(224,224),interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite("out1.jpg",crp)
            img=image.load_img("out1.jpg", target_size=(224,224))
            img=np.expand_dims(img, axis=0)
            result=iran_model.predict_proba(img)
            if result[0][0]==1.0 and dist<0.45:
                cv2.rectangle(frame, (left, top), (right,bottom), (0,255,0),3)
                y= top - 15 if top - 15>15 else top+15
                cv2.putText(frame, name, (left,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2) 
                gpio.output(12, gpio.HIGH)
                time.sleep(5)
                gpio.output(12, gpio.LOW)
            else:
                cv2.rectangle(frame, (left, top), (right,bottom), (0,0,255),3)
                y= top - 15 if top - 15>15 else top+15
                
                cv2.putText(frame, 'intruso', (left,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                cv2.putText(frame, 'Pressione "s" para acessar via senha', (100,460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)                 

        if name=="wands":
            dist=face_recognition.face_distance(encoding,wands_encod)
            crp= frame[int(boxes[0][0]):int(boxes[0][2]),int(boxes[0][3]):int(boxes[0][1])]
            crp= cv2.resize(crp,(224,224),interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite("out2.jpg",crp)
            img=image.load_img("out2.jpg", target_size=(224,224))
            img=np.expand_dims(img, axis=0)
            result=wands_model.predict_proba(img)
            if dist<0.45:
                cv2.rectangle(frame, (left, top), (right,bottom), (0,255,0),3)
                y= top - 15 if top - 15>15 else top+15
                cv2.putText(frame, name, (left,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2) 
                gpio.output(12, gpio.HIGH)
                time.sleep(5)
                gpio.output(12, gpio.LOW)            
            else:
                cv2.rectangle(frame, (left, top), (right,bottom), (0,0,255),3)
                y= top - 15 if top - 15>15 else top+15
                
                cv2.putText(frame, 'intruso', (left,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                cv2.putText(frame, 'Pressione "s" para acessar via senha', (100,460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2) 



            
    if ret==True:
        #out.write(frame)
        
        esca=0               
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.putText(frameout, 'Pressione "e" para sair', (100,460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
            cv2.imshow('frame',frameout)
            while cv2.waitKey(1) & 0xFF != ord('1'):
                print("digite...")
                if cv2.waitKey(1) & 0xFF == ord('e'):
                    esca+=1
                    break                
            while cv2.waitKey(1) & 0xFF != ord('x'):
                print("digite...")
                if cv2.waitKey(1) & 0xFF == ord('e'):
                    esca+=1
                    break                
            while cv2.waitKey(1) & 0xFF != ord('3'):
                print("digite...")
                if cv2.waitKey(1) & 0xFF == ord('e'):
                    esca+=1
                    break
            if esca==0:
                gpio.output(12, gpio.HIGH)
                time.sleep(5)
                gpio.output(12, gpio.LOW)
                print("liberado!!!")
            
    else:
        break

# Release everything if job is finished
cap.release()

cv2.destroyAllWindows()
