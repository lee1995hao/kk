import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure()
plt.imshow(train_images[1])##this is color
plt.subplot(1,2)
plt.show()

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(train_images[i],cmap=plt.cm.binary)##camp is gald

plt.show()

