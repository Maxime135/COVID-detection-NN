from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def PlotAccuracy(model):
  plt.figure("Accuracy")
  plt.plot(model.history['accuracy'],'ro', label='Training accuracy')
  plt.plot(model.history['val_accuracy'],'bo', label="Test accuracy")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.grid()
  plt.legend(loc='best')
  plt.show()


# Dataset :

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()


x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

num_classes = 10
input_shape = np.shape(x_train[0])

#on convertit les listes Y en vecteur

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

#Modele :

modelChiffre = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

modelChiffre.summary()

#Entrainement

batch_size_chiffre = 128
epochs_chiffre = 4

modelChiffre.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
modelChiffre.summary()

modelChiffreAccuracy = modelChiffre.fit(x_train, y_train, batch_size=batch_size_chiffre, epochs=epochs_chiffre, validation_split=0.1)


PlotAccuracy(modelChiffreAccuracy)

#Prediction

y_prediction = modelChiffre.predict(x_test)

### on converti y_prédiction d'un tableau vers une liste
y_prediction=y_prediction.tolist() 

### on choisi de garder seulement la classe avec la plus grande probabilité, pour un chiffre donné
print("\n\nComparaison modèle et réponses attendues :")
y_prediction=[L.index(max(L)) for L in y_prediction]
print(y_prediction)
print([L.index(max(L)) for L in y_test.tolist()])