from utils import load_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(feature, labels) = load_data()

x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.1)

categories = ['daisy','dandelion','kale'];

num_classes=len(categories);

input_layer = tf.keras.layers.Input([224, 224, 3])

conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu')(input_layer)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)

conv3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

conv4 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu')(pool3)
pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)

flt1 = tf.keras.layers.Flatten()(pool4)
dn1 = tf.keras.layers.Dense(512, activation='relu')(flt1)  # rectifying linear Unit
out = tf.keras.layers.Dense(num_classes, activation='softmax')(dn1)

model = tf.keras.Model(input_layer, out)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=20, epochs=3, validation_data=(x_test, y_test))

model.save('Identification Of Plants.h5')

# Plotting the loss and accuracy
plt.figure(figsize=(10, 4))

# Plot training & validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.show()
