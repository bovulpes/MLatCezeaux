# tensorflow2.pdf, p.124

import tensorflow as tf
import tensorflow_datasets as tfds

loader = tfds.load("cifar10", as_supervised=True)
train_ds, test_ds = loader["train"], loader["test"]

def normalize_resize(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    image = tf.image.resize(image, (28, 28))
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    return image, label 

train = train_ds.map(normalize_resize).cache().map(augment).shuffle(100).batch(64).repeat()
test = test_ds.map(normalize_resize).cache().batch(64)
