######################----------TRAIN DEEP LEARNING MODEL-----------------###################33

# STEP 1 - LOAD THE PREPROCESS DATA

import os
import numpy as np
import cv2
import gc

# load the preprocessed data
data = np.load('./dataset/data_preprocess.npz')

X = data['arr_0']
y = data['arr_1']

X.shape, y.shape

np.unique(y)

from sklearn.preprocessing import OneHotEncoder

onehot = OneHotEncoder()
y_onehot = onehot.fit_transform(y.reshape(-1,1))

y_array = y_onehot.toarray()

# Split the Data into Train and Test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y_array, test_size=0.2, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

y_train

cv2.imshow('a', x_train[-2])
cv2.waitKey()
cv2.destroyAllWindows()

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential

# convolution neural network

model = Sequential([
    layers.Conv2D(16, 3, padding='same', input_shape=(100, 100, 3), activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(192, activation='relu'),
    layers.Dense(28, activation='relu'),
    layers.Dense(4, activation='sigmoid')
])


# compiling CNN
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

# Training CNN
history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=30, epochs=20)



import pandas as pd
import matplotlib.pyplot as plt

history_df = pd.DataFrame(history.history)

history_df

history_df[['loss','val_loss']].plot(kind='line')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(list(range(10)),list(range(1,11)))
plt.show()

history_df[['accuracy','val_accuracy']].plot(kind='line')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(list(range(10)),list(range(1,11)))
plt.show()

# Save CNN Model
model.save('face_cnn_model')

onehot.categories_