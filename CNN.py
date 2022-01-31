import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = tf.reshape(train_images, [-1, 28, 28, 1])
test_images = tf.reshape(test_images, [-1, 28, 28, 1])
np.reshape(test_images[1],28*28)
def select_model(model_number):
    if model_number == 1:
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # layer 1
            keras.layers.MaxPool2D((2, 2)),  # layer 2
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation='softmax')])  # layer 3

    if model_number == 2:
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # layer 1 32output dim 3*3is the dim of 卷积窗口的宽度和高度
            keras.layers.MaxPool2D((2, 2)),  # layer 2 pool is plus covn 层的得分 （3，3）——》（2，2）
            keras.layers.Conv2D(64, (3, 3), activation='relu'),  # layer 3
            keras.layers.MaxPool2D((2, 2)),  # layer 4
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation='softmax')])  # layer 5

    if model_number == 3:
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # layer 1
            keras.layers.MaxPool2D((2, 2)),  # layer 2
            keras.layers.Conv2D(64, (3, 3), activation='relu'),  # layer 3
            keras.layers.Conv2D(64, (3, 3), activation='relu'),  # layer 4
            keras.layers.MaxPool2D((2, 2)),  # layer 5
            keras.layers.Conv2D(128, (3, 3), activation='relu'),  # layer 6
            keras.layers.Flatten(),    ##即返回一个一维数组
            keras.layers.Dense(10, activation='softmax')])  # layer 7

    return model



model = select_model(model_number = 1)

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)


model.fit(train_images,train_labels)

test_data = tf.reshape(test_images[0],shape=[-1,28,28,1])
model.predict(test_data)
plt.figure()
plt.imshow(test_images[1])##this is color
plt.subplot(1,2)
plt.show()