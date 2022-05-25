import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()

a = tf.compat.v1.get_variable("A", initializer=tf.constant(3, shape=[2]))
b = tf.compat.v1.get_variable("B", initializer=tf.constant(5, shape=[3]))

init_op = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    # initialize all of the variables in the session
    sess.run(init_op)
    # run the session to get the value of the variable
    a_out, b_out = sess.run([a, b])
    print('a = ', a_out)
    print('b = ', b_out)

saver = tf.compat.v1.train.Saver()
for i, var in enumerate(saver._var_list):
    print('Var {}: {}'.format(i, var))

with tf.compat.v1.Session() as sess:
    # initialize all of the variables in the session
    sess.run(init_op)

    # save the variable in the disk
    saved_path = saver.save(sess, './saved_variable')
    print('model saved in {}'.format(saved_path))

import os
for file in os.listdir('.'):
    if 'saved_variable' in file:
        print(file)

a_out = []
b_out = []
with tf.compat.v1.Session() as sess:
    # restore the saved vairable
    saver.restore(sess, './saved_variable')
    # print the loaded variable
    a_out, b_out = sess.run([a, b])
    print('a = ', a_out)
    print('b = ', b_out)

tf.compat.v1.reset_default_graph()
try:
    with tf.compat.v1.Session() as sess:
        # restore the saved vairable
        saver.restore(sess, './saved_variable')
        # print the loaded variable
        a_out, b_out = sess.run([a, b])
        print('a = ', a_out)
        print('b = ', b_out)
except Exception as e:
    print(str(e))

tf.compat.v1.reset_default_graph()

a = tf.compat.v1.get_variable("A", initializer=tf.constant(3, shape=[2]))
b = tf.compat.v1.get_variable("B", initializer=tf.constant(5, shape=[3]))

init_op = tf.compat.v1.global_variables_initializer()# create the graph

saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    # restore the saved vairable
    saver.restore(sess, './saved_variable')
    # print the loaded variable
    a_out, b_out = sess.run([a, b])
    print('a = ', a_out)
    print('b = ', b_out)

tf.compat.v1.reset_default_graph()

imported_graph = tf.compat.v1.train.import_meta_graph('saved_variable.meta')

for tensor in tf.compat.v1.get_default_graph().get_operations():
    print (tensor.name)

with tf.compat.v1.Session() as sess:
    # restore the saved vairable
    imported_graph.restore(sess, './saved_variable')
    # print the loaded variable
    a_out, b_out = sess.run(['A:0','B:0'])
    print('a = ', a_out)
    print('b = ', b_out)

