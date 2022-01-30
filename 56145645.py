import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images  = training_images / 255.0
test_images = test_images / 255.0

import matplotlib.pyplot as plt
##将图片转换为10*10
plt.imshow(training_images[0],cmap=plt.cm.binary)
training_images[0]
photo = cv2.resize(training_images[0],(10,10))
plt.imshow(photo,cmap='gray')
plt.show()
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])



model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)



import cv2


import tensorflow as tf
print('tf version:\n',tf.__version__)


import keras
print('keras version:\n',keras.__version__)

