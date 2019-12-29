#MODELO PADRAO DE CNN, UTILIZADA PARA, UMA REDE DESSA DEVE SER TREINADA PARA CADA USUARIO CADASTRADO NO SISTEMA.


# importando bibliotecas
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
import tensorflow as tf
  
  
img_width, img_height = 224, 224 #resolucao da imagem
  
train_data_dir = 'dataset_keras/train' #local das amostras de treino
validation_data_dir = 'dataset_keras/test' # local das amostras de teste
nb_train_samples = 800  #numero de amostras de treino
nb_validation_samples = 160  #numero de amostras de teste
epochs = 25 #epoca
batch_size = 16 #tamanho do lote
  
if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) #formato da imagem 3 canais rgb, 224 por 224
else: 
    input_shape = (img_width, img_height, 3) 
#arquitetura da rede cnn  
model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape = input_shape)) #32 kernels de 2 por 2
model.add(Activation('relu')) #funcao de ativacao relu
model.add(MaxPooling2D(pool_size =(2, 2))) #maxpooling de 2 por 2
  
model.add(Conv2D(32, (2, 2))) #32 kernels de 2 por 2
model.add(Activation('relu')) #funcao de ativcao relu
model.add(MaxPooling2D(pool_size =(2, 2))) #maxpooling 2 por 2
  
model.add(Conv2D(64, (2, 2))) #64 kernels de 2 por 2
model.add(Activation('relu')) #funcao de ativacao relu
model.add(MaxPooling2D(pool_size =(2, 2))) #maxpooling de 2 por 2
  
model.add(Flatten()) #concatenando os neuronios em uma unica camada densa
model.add(Dense(64)) #camada densa contendo 64 neuronios
model.add(Activation('relu')) #funcao de ativacao relu
model.add(Dropout(0.5)) #dropout de 50%
model.add(Dense(1)) #neuronio de saida
model.add(Activation('sigmoid')) #funcao de ativacao relu
  
model.compile(loss ='binary_crossentropy', 
                     optimizer ='RMSprop', 
                   metrics =['accuracy']) #optimizadores utilizados
  
train_datagen = ImageDataGenerator( 
                rescale = 1. / 255, 
                 shear_range = 0.2, 
                  zoom_range = 0.2, 
            horizontal_flip = True) #tratamentos iniciais na imagem para gerar os dados de treinamento
  
test_datagen = ImageDataGenerator(rescale = 1. / 255) 
  
train_generator = train_datagen.flow_from_directory(train_data_dir, 
                              target_size =(img_width, img_height), 
                     batch_size = batch_size, class_mode ='binary') 
  
validation_generator = test_datagen.flow_from_directory( 
                                    validation_data_dir, 
                   target_size =(img_width, img_height), 
          batch_size = batch_size, class_mode ='binary') 
  
model.fit_generator(train_generator, 
    steps_per_epoch = nb_train_samples // batch_size, 
    epochs = epochs, validation_data = validation_generator, 
    validation_steps = nb_validation_samples // batch_size) #comando par treinar a rede utilizando os dados fornecidos
  
model.save('henrique_saved.h5') #salvando o modelo treinado para ser utilizado para predicao
