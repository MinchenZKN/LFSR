import numpy as np
import h5py
import os
import sys
import file_io
import tensorflow as tf
import lf_tools
import matplotlib.pyplot as plt
import scipy
import random
from tfinterp import regular_nd




def next_batch(cv,disp,v,h,batch_size):

    ranges = disp.shape[2]
    order = random.sample(range(0,ranges),batch_size)
    cv = cv.transpose(3,0,1,2)
    disp = disp.transpose(2, 0, 1)
    v = v.transpose(4, 0, 1, 2, 3)
    h = h.transpose(4, 0, 1, 2, 3)

    return cv[order,:,:,:],disp[order,:,:],v[order,:,:,:,:],h[order,:,:,:,:]



def refocus_cross_tf(cv,disp,nviews,batch_size,grid_range,channels):
    nt = nviews
    ns = nviews


    height = cv.shape[1]
    width = cv.shape[2]
    # channels = cv.shape[3]
    range = height-1
    cv = tf.expand_dims(cv,1)
    cv_v = tf.tile(cv, (1, nt, 1, 1, 1))
    # cv_v = tf.transpose(cv_v, perm = [1, 0, 2, 3, 4])
    cv_h = tf.tile(tf.transpose(cv, perm = [0, 1, 3, 2, 4]), ( 1, nt, 1, 1, 1))
    # cv_h = tf.transpose(cv_h, perm =[1, 0, 2, 3, 4])

    disp = tf.expand_dims(disp,1)
    disp_v = tf.tile(disp, (1, nt, 1, 1))
    disp_h = tf.tile(tf.transpose(disp,perm = [0, 1, 3, 2]), (1, ns, 1, 1))
    # disp_v = tf.transpose(disp_v,perm=[1, 0, 2, 3])
    # disp_h = tf.transpose(disp_h,perm=[1, 0, 2, 3])
    # batch = disp.shape[0]
    batch = batch_size
    # disp_v = tf.nan_to_num(disp_v)
    # disp_h = tf.nan_to_num(disp_h)
    c = tf.constant((nt + 1) / 2 - 1)


    [p, q, v, u] = tf.meshgrid(tf.range(batch), tf.range(nt), tf.range(height), tf.range(width))
    pp = tf.cast(tf.transpose(p, perm=[1, 0, 2, 3]),tf.float32)
    qq = tf.cast(tf.transpose(q, perm=[1, 0, 2, 3]),tf.float32)
    vv = tf.cast(tf.transpose(v, perm=[1, 0, 2, 3]),tf.float32)
    uu = tf.cast(tf.transpose(u, perm=[1, 0, 2, 3]),tf.float32)

    y = tf.matmul((qq - c), (disp_v)) + vv
    x = tf.matmul((qq - c), (disp_h)) + vv

    points_v = [tf.cast(tf.range(batch),tf.float32), tf.cast(tf.range(nt),tf.float32),
                tf.cast(tf.range(height),tf.float32), tf.cast(tf.range(width),tf.float32)]
    points_h = [tf.range(batch), tf.range(ns), tf.range(height), tf.range(width)]

    # points_v[:] = tf.cast(points_v[:],tf.float32)

    a = tf.clip_by_value(tf.reshape(y, [-1]), 0, grid_range)
    b = tf.clip_by_value(tf.reshape(x, [-1]), 0, grid_range)

    xi_v = (tf.reshape(pp, [-1]), tf.reshape(qq, [-1]), a, tf.reshape(uu, [-1]))
    xi_h = (tf.reshape(pp, [-1]), tf.reshape(qq, [-1]), b, tf.reshape(uu, [-1]))

    stack_v = tf.zeros([batch, nt, height, width, channels])
    stack_h = tf.zeros([batch, ns, height, width, channels])

    # for i in range(channels):
    #     stack_v[:, :, :, :, i] = regular_nd(points=points_v, values=cv_v[:, :, :, :, i],
    #                                                        xi=xi_v).reshape((batch, nt, height, width))
    #     stack_h[:, :, :, :, i] = regular_nd(points=points_h, values=cv_h[:, :, :, :, i],
    #                                                        xi=xi_h).reshape((batch, nt, height, width))

    stack_v[:, :, :, :, 0] = regular_nd(points=points_v, values=cv_v[:, :, :, :, 0], xi=xi_v)



    return stack_h, stack_v

