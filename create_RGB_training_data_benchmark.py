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
# from skimage.color import rgb2hsv, hsv2rgb
from libs.convert_colorspace import rgb2YCbCr, YCbCr2rgb
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from scipy.ndimage import gaussian_filter
# python tools for our lf database
import file_io
# additional light field tools
import lf_tools
import matplotlib.pyplot as plt
# OUTPUT CONFIGURATION

# patch size. patches of this size will be extracted and stored
# must remain fixed, hard-coded in NN

scale = 2

px_LR = 48
py_LR = 48
px = int(px_LR * scale)
py = int(py_LR * scale)
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
sx_LR = 16
sy_LR = 16
sx = int(sx_LR * scale)
sy = int(sy_LR * scale)



# output file to write to
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! careful: overwrite mode !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# previous training data will be erased.

training_data_dir = "/home/mz/"
training_data_filename = 'lf_patch_benchmark_rgb_sr.hdf5'
file = h5py.File(training_data_dir + training_data_filename, 'w')

#
# data_folders = ( ( "training", "boxes" ), )
# data_folders = data_folders_base + data_folders_add
# data_source_LR = "/home/z/PycharmProjects/SR/full_data_256_9/"
data_source = "/home/mz/HD data/SR data backups/full_data_512/"
# data_folders = os.listdir(data_source)
data_folders = []
data_folders.append('dishes')
# data_folders.append('greek')
# data_folders.append('tower')
# data_folders.append('antinous')
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

# EPI patches, nviews x patch size x patch size x channels
# horizontal and vertical direction (to get crosshair)
dset_v = file.create_dataset( 'stacks_v', (nviews, py_LR, px_LR, 3, 1),
                              chunks = (nviews, py_LR, px_LR, 3, 1),
                              maxshape = (nviews, py_LR, px_LR, 3, None))

dset_h = file.create_dataset('stacks_h', (nviews, py_LR, px_LR, 3, 1),
                             chunks=(nviews, py_LR, px_LR, 3, 1),
                             maxshape=(nviews, py_LR, px_LR, 3, None))

# dataset for correcponsing center view patch (to train joint upsampling)
# ideally, would want to reconstruct full 4D LF patch, but probably too memory-intensive
# keep for future work
dset_cv = file.create_dataset('cv', (py, px, 3, 1),
                              chunks=(py, px, 3, 1),
                              maxshape=(py, px, 3, None))



#
# loop over all datasets, write out each dataset in patches
# to feed to autoencoder in random order
#
index = 0
for lf_name in data_folders:

    data_folder = os.path.join(data_source, lf_name)

    LF = file_io.read_lightfield(data_folder)
    LF = LF.astype(np.float32)

    LF_LR = np.zeros((LF.shape[0],LF.shape[1],int(LF.shape[2]/scale),
                    int(LF.shape[3]/scale),int(LF.shape[4])),np.float32)
    #
    # disp = file_io.read_disparity(data_folder)
    # disp_gt = np.array(disp[0])
    # disp_gt = np.flip(disp_gt, 0)
    cv_gt = lf_tools.cv(LF)

    for v in range(0, nviews):
        for h in range(0, nviews):
            LF[v, h, :, :, :] = gaussian_filter(LF[v, h, :, :, :], sigma=0.5, truncate=2)
            LF_LR[v, h, :, :, :] = LF[v, h, 0:LF.shape[2] - 1:scale, 0:LF.shape[3] - 1:scale, :]


    # lf_tools.save_image(training_data_dir + 'input' + lf_name, cv_gt)

    # maybe we need those, probably not.
    # param_dict = file_io.read_parameters(data_folder)

    # write out one individual light field
    # block count
    cx_LR = np.int32((LF_LR.shape[3] - px_LR) / sx_LR) + 1
    cy_LR = np.int32((LF_LR.shape[2] - py_LR) / sy_LR) + 1
    cx_HR = np.int32((LF.shape[3] - px) / sx) + 1
    cy_HR = np.int32((LF.shape[2] - py) / sy) + 1


    for by in np.arange(0, cy_LR):
        sys.stdout.write('.')
        sys.stdout.flush()

        for bx in np.arange(0, cx_LR):
            x_LR = bx * sx_LR
            y_LR = by * sx_LR

            x = bx * sx
            y = by * sx
            # extract data
            (stack_v, stack_h) = lf_tools.epi_stacks(LF_LR, y_LR, x_LR, py_LR, px_LR)
            # make sure the direction of the view shift is the first spatial dimension
            stack_h = np.transpose(stack_h, (0, 2, 1, 3))

            cv = cv_gt[y:y + py, x:x + px]

            # write to respective HDF5 datasets
            dset_v.resize(index + 1, 4)
            dset_v[:, :, :, :, index] = stack_v

            dset_h.resize(index + 1, 4)
            dset_h[:, :, :, :, index] = stack_h

            dset_cv.resize(index + 1, 3)
            dset_cv[:, :, :, index] = cv

            # next patch
            index = index + 1

    # next dataset
    print(' done.')

