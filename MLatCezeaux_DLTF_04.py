import tensorflow as tf
import numpy as np

a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)

print(type(a), a)
print(type(ta), ta)

