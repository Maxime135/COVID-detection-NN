from tensorflow import keras
import tensorflow
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import *


def PlotAccuracy(model):
  plt.figure("Accuracy")
  plt.plot(model.history['accuracy'],'ro', label='Training accuracy')
  plt.plot(model.history['val_accuracy'],'bo', label="Test accuracy")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.grid()
  plt.legend(loc='best')
  plt.show()

#Dataset :

image_size = (340,340)
batch_size = 32
num_classes = 3


#os.chdir("/Users/maximeboulanger/Documents/CNN Covid19 xray/covid dataset/Covid19-dataset")
os.chdir("C:\\Users\\maxim\\Documents\\IA\\COVID Github\\covid dataset\\Covid19-dataset")


train_ds = keras.preprocessing.image_dataset_from_directory(
    "train",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical',
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    "train",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical',
)

# 0 : Covid
# 1 : Normal
# 2 : Pneumonia

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(str(labels[i]))
        plt.axis("off")
plt.show()


# Dataaugmentation :

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

# Standardizing the data :

""" augmented_train_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y)) """

# Configure the dataset for performance
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


# Modele :

model = keras.Sequential(
    [
        keras.Input(shape=image_size+(3,)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

epochs_image = 10
batch_size_image = 150

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

modelAccuracy = model.fit(train_ds, batch_size=batch_size_image, epochs=epochs_image, validation_data=val_ds)
PlotAccuracy(modelAccuracy)


# Test the NN on new data :

img = keras.preprocessing.image.load_img(
    askopenfilename(), target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]

print("Covid | Normal | Viral pneumonia")
print(score)