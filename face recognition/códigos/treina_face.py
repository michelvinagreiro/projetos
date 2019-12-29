'''
Created on 12 de set de 2019

@author: michel
'''
import numpy as np
import face_recognition
import cv2
import pickle
kencoding=[]#pilha de armazenamento dos vetores
knames=[]# pilha de armazenamento dos nomes


for iran in range(3):
    a=cv2.imread("iranilson/img"+str(iran+30)+".jpg",3)# dentre as imagens do usuario le as imagens img30.jpg, img31.jpg e img32.jpg
        
    boxes=face_recognition.face_locations(a,model="cnn")#localiza as faces dentro da imagem
    encodings=face_recognition.face_encodings(a,boxes)#apos localizada a face, gera o vetor de caracteristicas
    for encoding in encodings:
        kencoding.append(encoding)#guarda o vetor dentro da pilha
        knames.append("iranilson")#guarda o nome do usuario dentro da pilha
        print("processando iranilson img"+str(iran))#mostra o status para o programador

for hen in range(3):    
    a=cv2.imread("henrique/img"+str(hen+155)+".jpg",3)#img155.jpg, img156.jpg e img157.jpg
    boxes=face_recognition.face_locations(a,model="cnn")
    encodings=face_recognition.face_encodings(a,boxes)
    for encoding in encodings:
        print(boxes)
        kencoding.append(encoding)
        knames.append("henrique")
    print("processando henrique img"+str(hen))

for wand in range(3):    
    a=cv2.imread("wands/img"+str(wand+670)+".jpg",3)#img670.jpg, img671.jpg e img672.jpg
    boxes=face_recognition.face_locations(a,model="cnn")
    encodings=face_recognition.face_encodings(a,boxes)
    for encoding in encodings:
        kencoding.append(encoding)
        knames.append("wands")
    print("processando wands img"+ str(wand))

for y in range(200):# no fei database as imagens sao separadas em 1a.jpg e 1b.jpg, 2a.jpg e 2b.jpg, 3a.jpg e 3b.jpg,....faces neutras e sorridentes
    a=cv2.imread("FEI_frontal/"+str(y+1)+"a.jpg",3)#le a face neutra
    print("imagem"+str(y+1)+"a")
    boxes=face_recognition.face_locations(a,model="cnn")
    encodings=face_recognition.face_encodings(a,boxes)
    for encoding in encodings:
        print(boxes)    
        kencoding.append(encoding)
        knames.append("intruso")
        print("processando Intruso_fei")
    a=cv2.imread("FEI_frontal/"+str(y+1)+"b.jpg",3)#le a face sorridente
    print("imagem"+str(y+1)+"b")
    boxes=face_recognition.face_locations(a,model="cnn")
    encodings=face_recognition.face_encodings(a,boxes)
    for encoding in encodings:
        print(boxes)    
        kencoding.append(encoding)
        knames.append("intruso")
        print("processando Intruso_fei")

data={"encodings":kencoding,"names":knames}#junta as duas pilhas em uma unica matriz
f=open("enc_TCC.pickle","wb")
f.write(pickle.dumps(data))# salva em formato de arquivo
f.close()        
print("terminou!")  
  
