'''
Created on 12 de set de 2019

@author: michel
'''
import numpy as np
import face_recognition
import cv2
import pickle
kencoding=[]
knames=[]


for iran in range(3):
    a=cv2.imread("iranilson/img"+str(iran+30)+".jpg",3)
        
    boxes=face_recognition.face_locations(a,model="cnn")
    encodings=face_recognition.face_encodings(a,boxes)
    for encoding in encodings:
        kencoding.append(encoding)
        knames.append("iranilson")
        print("processando iranilson img"+str(iran))

for hen in range(3):    
    a=cv2.imread("henrique/img"+str(hen+155)+".jpg",3)
    boxes=face_recognition.face_locations(a,model="cnn")
    encodings=face_recognition.face_encodings(a,boxes)
    for encoding in encodings:
        print(boxes)
        kencoding.append(encoding)
        knames.append("henrique")
    print("processando henrique img"+str(hen))

for wand in range(3):    
    a=cv2.imread("wands/img"+str(wand+670)+".jpg",3)
    boxes=face_recognition.face_locations(a,model="cnn")
    encodings=face_recognition.face_encodings(a,boxes)
    for encoding in encodings:
        kencoding.append(encoding)
        knames.append("wands")
    print("processando wands img"+ str(wand))

for y in range(200):
    a=cv2.imread("FEI_frontal/"+str(y+1)+"a.jpg",3)
    print("imagem"+str(y+1)+"a")
    boxes=face_recognition.face_locations(a,model="cnn")
    encodings=face_recognition.face_encodings(a,boxes)
    for encoding in encodings:
        print(boxes)    
        kencoding.append(encoding)
        knames.append("intruso")
        print("processando Intruso_fei")
    a=cv2.imread("FEI_frontal/"+str(y+1)+"b.jpg",3)
    print("imagem"+str(y+1)+"b")
    boxes=face_recognition.face_locations(a,model="cnn")
    encodings=face_recognition.face_encodings(a,boxes)
    for encoding in encodings:
        print(boxes)    
        kencoding.append(encoding)
        knames.append("intruso")
        print("processando Intruso_fei")

data={"encodings":kencoding,"names":knames}
f=open("enc_TCC.pickle","wb")
f.write(pickle.dumps(data))
f.close()        
print("terminou!")  
  
