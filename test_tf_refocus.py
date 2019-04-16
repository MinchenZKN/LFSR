import numpy as np
import h5py
import os
import sys
import file_io
import tensorflow as tf
import lf_tools
import matplotlib.pyplot as plt
import scipy
from tf_test_libs import refocus_cross_tf, next_batch


file_name = '/home/mz/HD data/tf_test.hdf5'
f = h5py.File(file_name,'r+')
cv = f['cv']
disp = f['disp']
stack_v = f['stacks_v']
stack_h = f['stacks_h']
batch_size = 5

cv_in = tf.placeholder(tf.float32, shape=[None, 512,512,3])
disp_in = tf.placeholder(tf.float32, shape=[None, 512,512])
stack_v_gt = tf.placeholder(tf.float32, shape=[None, 9,512,512,3])
stack_h_gt = tf.placeholder(tf.float32, shape=[None, 9,512,512,3])

stack_h_0,stack_v_0 = refocus_cross_tf(cv_in,disp_in,9,batch_size,511,3)


W1 = tf.Variable(tf.zeros([3,3,3,3,64]))
stack_v_1 = tf.nn.relu(tf.nn.conv3d(stack_v_0,W1,[1,1,1,1,1],padding='SAME'))
stack_h_1 = tf.nn.relu(tf.nn.conv3d(stack_h_0,W1,[1,1,1,1,1],padding='SAME'))


W2 = tf.Variable(tf.zeros([3,3,3,64,3]))
stack_v_2 = tf.nn.relu(tf.nn.conv3d(stack_v_1,W2,[1,1,1,1,1],padding='SAME'))
stack_h_2 = tf.nn.relu(tf.nn.conv3d(stack_h_1,W2,[1,1,1,1,1],padding='SAME'))

loss = tf.losses.mean_squared_error(stack_v_2,stack_v_gt) + tf.losses.mean_squared_error(stack_h_2,stack_h_gt)
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)




for i in range(1000):
    b_cv, b_disp, b_v, b_h = next_batch(cv,disp,stack_v,stack_h,batch_size)
    loss_value = sess.run(train_step, feed_dict={cv_in: b_cv, disp_in: b_disp, stack_v_gt:b_v, stack_h_gt:b_h})
    print(loss_value)