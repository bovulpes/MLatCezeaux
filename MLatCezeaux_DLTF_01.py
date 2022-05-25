import tensorflow as tf
print("TF Version:",tf.__version__)
print("eager execution:",tf.executing_eagerly())

# 01
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
print(c)

# 02, tensorflow2.pdf p.25
print("1 + 2 + 3 + 4 =", tf.reduce_sum([1, 2, 3, 4]))

# 03, tensorflow2.pdf p.30
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print("v.value():", v.value())
print("")
print("v.numpy():", v.numpy())
print("")

# 04, tensorflow2.pdf p.37
def func1():
  a = tf.constant([[10,10],[11.,1.]])
  b = tf.constant([[1.,0.],[0.,1.]])
  c = tf.matmul(a, b)
  return c
print(func1().numpy())

# 05, idem
# defines a graph and performs a session execution (like in TF1)
@tf.function
def func2():
  a = tf.constant([[10,10],[11.,1.]])
  b = tf.constant([[1.,0.],[0.,1.]])
  c = tf.matmul(a, b)
  return c
print(func2().numpy())

# 06, tensorflow2.pdf p.51
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w
  grad = tape.gradient(loss, w)
  print("grad:",grad)

# 07, idem
x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = 4 * x * x
  dy_dx = g.gradient(y, x)
  print("dy_dx:",dy_dx)

# 08, tensorflow2.pdf p.52
x = tf.constant(4.0)
with tf.GradientTape() as t1: 
  with tf.GradientTape() as t2:
    t1.watch(x)
    t2.watch(x)
    z = x * x * x
  dz_dx = t2.gradient(z, x)
d2z_dx2 = t1.gradient(dz_dx, x)
print("First dz_dx: ",dz_dx)
print("Second d2z_dx2:",d2z_dx2)

# 09, idem
x = tf.Variable(4.0)
with tf.GradientTape() as t1:
  with tf.GradientTape() as t2:
    z = x * x * x
  dz_dx = t2.gradient(z, x)
d2z_dx2 = t1.gradient(dz_dx, x)
print("First dz_dx: ",dz_dx)
print("Second d2z_dx2:",d2z_dx2)

# 10, tensorflow2.pdf p.53
x = tf.ones((3, 3))
with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  print("y:",y)
  z = tf.multiply(y, y)
  print("z:",z)
  z = tf.multiply(z, y)
  print("z:",z)
# the derivative of z with respect to y
dz_dy = t.gradient(z, y)
print("dz_dy:",dz_dy)

# 11, tensorflow2.pdf p.54
x = tf.ones((3, 3))
with tf.GradientTape(persistent=True) as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  print("y:",y)
  w = tf.multiply(y, y)
  print("w:",w)
  z = tf.multiply(y, y)
  print("z:",z)
  z = tf.multiply(z, y)
  print("z:",z)
# the derivative of z with respect to y
dz_dy = t.gradient(z, y)
print("dz_dy:",dz_dy)
dw_dy = t.gradient(w, y)
print("dw_dy:",dw_dy)
