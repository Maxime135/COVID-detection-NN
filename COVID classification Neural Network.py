import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
import random as rd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

#chemins d'accès des données (à rogner et changer résolution)

pathTestCovid ="C:\\Users\\maxim\\Documents\\IA\\covid dataset\\Covid19-dataset\\test\\Covid"
pathTestNormal ="C:\\Users\\maxim\\Documents\\IA\\covid dataset\\Covid19-dataset\\test\\Normal"
pathTestPneumonia ="C:\\Users\\maxim\\Documents\\IA\\covid dataset\\Covid19-dataset\\test\\Viral Pneumonia"

pathTrainCovid ="C:\\Users\\maxim\\Documents\\IA\\covid dataset\\Covid19-dataset\\train\\Covid"
pathTrainNormal ="C:\\Users\\maxim\\Documents\\IA\\covid dataset\\Covid19-dataset\\train\\Normal"
pathTrainPneumonia ="C:\\Users\\maxim\\Documents\\IA\\covid dataset\\Covid19-dataset\\train\\Viral Pneumonia"


os.chdir(pathTrainPneumonia)


# minn,minp,mind = np.shape(plt.imread(os.listdir()[0]))  #dimensions première image en référence
# 
# 
# for image in os.listdir():
#     n,p,d = np.shape(plt.imread(image))
#     print(n,p)
#     if n<minn and p<minp :
#         minn,minp=n,p
#         print(image)
# print(minn,minp)

# il faut passer les images en N&B et toutes les redimensionner en 340x340

# for imageName in os.listdir():
#     image = cv2.imread(imageName)
#     imageRes = cv2.resize(image,dsize=(340,340),interpolation = cv2.INTER_LINEAR)
#     plt.imsave("b&w_"+imageName[:-4],imageRes)

def RGBtoBW(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    # plt.imshow(imgGray, cmap='gray')
    # plt.show()
    return(imgGray)

# for imageName in os.listdir():
#     image = plt.imread(imageName)
#     plt.imsave("Ab&w_"+imageName[:-4],RGBtoBW(image))


## Préparation des données 

nb_classes = 3

# covid : 0
# normal : 1
# pneumonie : 2

input_shape = (340,340,3)

#banque de données mélangées :

X,Y=[],[] #listes contenant toutes les données
Lpath = [pathTrainCovid,pathTrainNormal,pathTrainPneumonia,pathTestCovid,pathTestNormal,pathTestPneumonia]

for path in Lpath:
    for imageName in os.listdir():
        # X.append(RGBtoBW(plt.imread(imageName))) #elements de X sont des images (plt.imread)
        X.append(plt.imread(imageName))
        if path==pathTrainCovid or path==pathTestCovid: #elements de Y sont des vecteurs
            Y.append(0)
        elif path==pathTrainNormal or path==pathTestNormal:
            Y.append(1)
        else:
            Y.append(2)

# X = np.array(X)

Y = keras.utils.to_categorical(np.array(Y))

# Make sure images have shape (340, 340, 1)
# X = np.expand_dims(X,-1)
print(np.shape(X[0]))

#listes d'entrainement (100) et de test (56)


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
#x_train = np.expand_dims(x_train, -1)
#x_test = np.expand_dims(x_test, -1)


#x_train,y_train,x_test,y_test=[],[],[],[]
#
#LAlreadyChosen = []
#while len(x_train)<150:
#    n = rd.randint(0,155)
#    if n not in LAlreadyChosen:
#        x_train.append(X[n])
#        y_train.append(Y[n])
#        LAlreadyChosen.append(n)
#while len(x_test)<6:
#    n = rd.randint(0,155)
#    if n not in LAlreadyChosen:
#        x_test.append(X[n])
#        y_test.append(Y[n])
#        LAlreadyChosen.append(n)
#
#x_train =np.array(x_train)
#x_test = np.array(x_test)

""" x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

#x_train = x_train.(x_train.shape[0], 340, 340, 1))
x_train = np.expand_dims(x_train, -1)
y_train = np.array(y_train)

#x_test = x_test.reshape((x_test.shape[0], 340, 340, 1))
x_test = np.expand_dims(x_test, -1)
y_test = np.array(y_test) """

#x_train = np.expand_dims(x_train, -1)
#x_test = np.expand_dims(x_test, -1)

## Achitecture réseau de neurones

modelRadio = keras.Sequential(
    [
        # keras.Input(shape=input_shape),
        # layers.Conv2D(16,kernel_size=(3,3),activation="softmax"),
        # # layers.Conv2D(16,kernel_size=(3,3),activation="relu"),
        # layers.MaxPooling2D(pool_size=(2,2)),
        # # layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
        # layers.Conv2D(32,kernel_size=(3,3),activation="softmax"),
        # layers.MaxPooling2D(pool_size=(2,2)),
        # # layers.Conv2D(128,kernel_size=(3,3),activation="relu"),
        # # layers.MaxPooling2D(pool_size=(2,2)),
        # layers.Flatten(),
        # layers.Dropout(0.7),
        # layers.Dense(nb_classes,activation="softmax"),
        
        keras.Input(shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(16,kernel_size=(3,3),padding='same',activation="relu"),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Dropout(0.35),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(nb_classes,activation="softmax"),
    ]
) 


""" model = keras.Sequential()

#### Input Layer ####
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), padding='same',
                 activation='relu', input_shape=input_shape))

#### Convolutional Layers ####
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))  # Pooling
model.add(layers.Dropout(0.2)) # Dropout

model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(512, (5,5), padding='same', activation='relu'))
model.add(layers.Conv2D(512, (5,5), activation='relu'))
model.add(layers.MaxPooling2D((4,4)))
model.add(layers.Dropout(0.2))

#### Fully-Connected Layer ####
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(nb_classes, activation='softmax')) """


modelRadio.summary()

## Entrainement:

def PlotAccuracy(model):
    plt.plot(model.history['accuracy'],'r', label='Training accuracy')
    plt.plot(model.history['val_accuracy'],'b', label="Test accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.ylim(0,1)
    plt.legend(loc='best')
    plt.show()


batch_size_radio = 2
epochs_radio = 3

modelRadio.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

modelRadioAccuracy = modelRadio.fit(x_train,y_train,batch_size=batch_size_radio, epochs=epochs_radio, validation_split=0.2)


PlotAccuracy(modelRadioAccuracy)

## Prediction :
y_prediction = modelRadio.predict(x_test)
print(y_prediction)
print(y_test)