import tensorflow as tf

tf.compat.v1.disable_eager_execution()

print("TF Version:",tf.__version__)
print("eager execution:",tf.executing_eagerly())

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

print(c)
