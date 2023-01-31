# Code made by Sergio Andrés Díaz Ariza
# date: 23/01/2023
# MIT Liscense

import tensorflow as tf

print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, MaxPooling2D

# Create a Instance of sequential model with 3 layers
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(16, activation='relu', name='layer_1'),  #
    Dense(16, activation='relu', name='layer_2'),  # Two first layers with activation radial and 16 units
    Dense(10, activation='softmax', name='layer_3')
])

# Print the weights of the model
print("\n")
print(
    model.weights)  # Maybe appear a Error Because you need initialize the weights and must specify the input, add flatten layer
print("\n")

# Print the summary of the model
print(model.summary())
print("\n")

# CREATE A CONVOLUTIONAL MODEL AND POOLING LAYER

model2 = Sequential([
    Conv2D(16, (3, 3), padding='SAME', strides=2, activation='relu', input_shape=(28, 28, 1),
           data_format='channels_last'),
    MaxPooling2D((3, 3), data_format='channels_last'),
    Flatten(),
    Dense(10, activation='softmax')
])

print(model2.summary())
print('\n')

# COMPILE METHOD
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mae = tf.keras.metrics.MeanAbsoluteError()

model2.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[acc, mae])

# Print the resulting compile
print(model2.optimizer.lr)
print(model2.loss)
print(model2.metrics)
