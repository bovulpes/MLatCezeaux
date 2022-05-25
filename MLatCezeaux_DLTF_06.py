import tensorflow as tf

x = tf.constant(3.0)

with tf.GradientTape() as g:
  g.watch(x)
  y = x * x

dy_dx = g.gradient(y, x)
print(dy_dx)

with tf.GradientTape() as g2:
  g2.watch(x)
  with tf.GradientTape() as g1:
    g1.watch(x)
    y = x * x
  dy_dx = g1.gradient(y, x)  # dy_dx = 2 * x
d2y_dx2 = g2.gradient(dy_dx, x)  # d2y_dx2 = 2
print(dy_dx)
print(d2y_dx2)

