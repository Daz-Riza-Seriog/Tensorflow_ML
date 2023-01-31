# Code made by Sergio Andrés Díaz Ariza
# date: 24/01/2023
# MIT Liscense

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import timeit

start = timeit.default_timer()
# sns.set()

# BUILD THE MODEL
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(10, activation='softmax')
])

# COMPILE THE METHOD
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=[acc, mae])

## FIT THE MODE
# Install the Data-set from Keras library

fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()

# look up the data set shape
print(train_images.shape)
print(train_labels)

# Define the labels
labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# Rescale the image values to 0 and 1
train_images = train_images / 255.
test_images = test_images / 255.

# Display one of this images
i = 0
img = train_images[i, :, :]
plt.figure(1)
plt.imshow(img)
# plt.show()
print(f"label: {labels[train_labels[i]]}")

# Fit the model or Train model
history = model.fit(train_images, train_labels, epochs=2, batch_size=256)

# Load the history into a Pandas Dataframe
df = pd.DataFrame(history.history)
print('\n')
print(df.head())

# Make a plot of the loss
loss_plot = df.plot(y='loss', title='Loss vs. Epochs', legend=False)
loss_plot.set(xlabel="Epochs", ylabel="Loss")

mean_abs_err = df.plot(y='mean_absolute_error', title='MAE vs. Epochs', legend=False)
mean_abs_err.set(xlabel="Epochs", ylabel="Mean Absolute Error")

# Evaluate and Predict Methods

# Evaluate the method
test_loss,test_accuarcy,test_mae = model.evaluate(test_images,test_labels)

# Make prediction of te model

# Choose a random image-In this case we use a only image that we choise
random_inx = np.random.choice(test_images.shape[0])
inx=30
test_image = test_images[inx]
plt.imshow(test_image)
plt.show()
print(f"Label: {labels[test_labels[inx]]}")

# Get the model prediction

predictions = model.predict(test_image[np.newaxis,...,np.newaxis]) # We put a dummy because you need enter a object with same dimension [np.newaxis,...,np.newaxis]
print(f"Model Prediction: {labels[np.argmax(predictions)]}")

# Predict a Batch of the labels simultaneously
random_imx = np.random.choice(test_images.shape[0],size=10)
test_image_ = test_images[random_imx]
prediction = model.predict(test_image_[...,np.newaxis])

cnt = 0
for i in prediction:
  print(f"Model Prediction: {labels[np.argmax(prediction[cnt])]}")
  cnt += 1


plt.show()
stop = timeit.default_timer()
print('Time: ', stop - start)
