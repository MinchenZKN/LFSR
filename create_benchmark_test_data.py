#!/usr/bin/python3
#
# read a bunch of source light fields and write out
# training data for our autoencoder in useful chunks
#
# pre-preparation is necessary as the training data
# will be fed to the trainer in random order, and keeping
# several light fields in memory is impractical.
#
# WARNING: store data on an SSD drive, otherwise randomly
# assembing a bunch of patches for training will
# take ages.
#
# (c) Bastian Goldluecke, Uni Konstanz
# bastian.goldluecke@uni.kn
# License: Creative Commons CC BY-SA 4.0
#

from queue import Queue
import time
import code
import os
import sys
import h5py

import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from scipy.ndimage import gaussian_filter
# python tools for our lf database
import file_io
# additional light field tools
import lf_tools
import matplotlib.pyplot as plt
# OUTPUT CONFIGURATION

# patch size. patches of this size will be extracted and stored
# must remain fixed, hard-coded in NN
s4 = 4
s2 = 2

px_LR_s4 = 48
py_LR_s4 = 48
px_LR_s2 = int(px_LR_s4 * s2)
py_LR_s2 = int(py_LR_s4 * s2)
px = int(px_LR_s4 * s4)
py = int(py_LR_s4 * s4)
# number of views in H/V/ direction
# input data must match this.
# nviews_LR = 5
nviews = 9
channel = 3

# block step size. this is only 16, as we keep only the center 16x16 block
# of each decoded patch (reason: reconstruction quality will probably strongly
# degrade towards the boundaries).
#
# TODO: test whether the block step can be decreased during decoding for speedup.
#
sx_LR_s4 = 16
sy_LR_s4 = 16
sx_LR_s2 = int(sx_LR_s4 * s2)
sy_LR_s2 = int(sx_LR_s4 * s2)
sx = int(sx_LR_s4 * s4)
sy = int(sy_LR_s4 * s4)

# output file to write to
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! careful: overwrite mode !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# previous training data will be erased.

test_data_dir = "H:\\testData\\super_resolution\\benchmark\\not seen\\"
test_data_filename = 'lf_test_benchmark_'
#
# data_folders = ( ( "training", "boxes" ), )
# data_folders = data_folders_base + data_folders_add
data_source = "E:\\MATLAB\\Benchmark_dataset\\data_only_benchmark\\"
# data_folders = os.listdir(data_source)
data_folders = []
# data_folders.append('dishes')
# data_folders.append('greek')
# data_folders.append('tower')
data_folders.append('antinous')
# data_folders.append('boardgames')
# data_folders.append('boxes')
# data_folders.append('cotton')
# data_folders.append('dino')
# data_folders.append('kitchen')
# data_folders.append('medieval2')
# data_folders.append('museum')
# data_folders.append('pens')
# data_folders.append('pillows')
# data_folders.append('platonic')
# data_folders.append('rosemary')
# data_folders.append('sideboard')
# data_folders.append('table')
# data_folders.append('tomb')
# data_folders.append('town')
# data_folders.append('vinyl')
data_folders.append('herbs')
data_folders.append('bicycle')
data_folders.append('bedroom')
data_folders.append('origami')

#
# loop over all datasets, write out each dataset in patches
# to feed to autoencoder in random order
#
index = 0
for lf_name in data_folders:
    file = h5py.File(test_data_dir + test_data_filename + lf_name + '.hdf5', 'w')
    data_folder = os.path.join(data_source, lf_name)
    LF = file_io.read_lightfield(data_folder)
    # LF_LR = LF_LR.astype(np.float32)
    LF = LF.astype(np.float32)
    cv_gt = lf_tools.cv(LF)

    disp = file_io.read_disparity(data_folder)
    disp_gt = np.array(disp[0])
    disp_gt = np.flip(disp_gt, 0)

    dset_LF = file.create_dataset('LF', data=LF)
    dset_disp = file.create_dataset('disp', data=disp_gt)

    # next dataset
    print(' done.')

