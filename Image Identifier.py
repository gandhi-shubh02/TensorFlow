import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math

tfds.disable_progress_bar()
# importing the fashion MNIST dataset provided by tensorflow
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
features_train, features_test = dataset['train'], dataset['test']
# 60k images in features_train
# 10k images in features_test
fashion_types = ['T-shit/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                 'Ankle boot']
# Each image is one of the following fashion types
num_of_train = metadata.splits['train'].num_examples
num_of_test = metadata.splits['test'].num_examples


def normalize(images, labels):  # value of each pixel is from 0 to 255
    # function makes it to 0 or 1
    images = tf.cast(images, tf.float32)  # cast() converts the datatype of the dataset
    images = images / 255
    return images, labels


features_train = features_train.map(normalize)  # map itrates all images in the dataset
features_test = features_train.map(normalize)  # cache() keeps data in ram to make trainning faster
l0=tf.keras.layers.Conv2D(32,(3,3),padding='same',activation=tf.nn.relu)
l1=tf.keras.layers.MaxPooling2D((2,2),strides=2)
l2=tf.keras.layers.Conv2D(64,(3,3),padding='same',activation=tf.nn.relu)
l3=tf.keras.layers.MaxPooling2D((2,2),strides=2)
l4 = tf.keras.layers.Flatten()  # converts a 2d image(28x28 pixels) into a 1d array
l5 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
l6 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
model = tf.keras.Sequential([l0, l1, l2,l3,l4,l5,l6])  # 10 beacuse 10 probabilities of output
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

features_train = features_train.cache().repeat().shuffle(num_of_train).batch(
    32)  # dataset.batch(32) tells model.fit to use batches of 32 images and labels when updating the model variables.
features_test = features_test.cache().repeat().shuffle(num_of_test).batch(32)
model.fit(features_train, epochs=10, steps_per_epoch=math.ceil(num_of_train / 32))
# softmax returns answer in probability
test_loss, test_accuracy = model.evaluate(features_test, steps=math.ceil(num_of_test / 32))
print('test acc', test_accuracy)
for test_images, test_labels in features_test.take(1):  # predictions
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
print(predictions[0])
