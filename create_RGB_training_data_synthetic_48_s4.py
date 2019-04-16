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
from skimage.color import rgb2hsv, hsv2rgb
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

training_data_dir = "/home/mz/HD data/SR data backups/Data_Cross_48_LF_s4/"
training_data_filename = 'lf_patch_synthetic_rgb_sr_s4_1.hdf5'
# inner_folder = 'sequence'
# flow = ['000000', '000001']
file = h5py.File( training_data_dir + training_data_filename, 'w' )

data_source = "/home/mz/HD data/Data_Cross_1024/"
# data_source = "/home/mz/HD data/test brightness/"
data_folders = os.listdir(data_source)
# take only part
# data_folders = data_folders[0:20]

# EPI patches, nviews x patch size x patch size x channels
# horizontal and vertical direction (to get crosshair)
dset_v_LR_s4 = file.create_dataset( 'stacks_v_LR_s4', (nviews, py_LR_s4, px_LR_s4, 3, 1),
                              chunks = (nviews, py_LR_s4, px_LR_s4, 3, 1),
                              maxshape = (nviews, py_LR_s4, px_LR_s4, 3, None))

dset_h_LR_s4 = file.create_dataset('stacks_h_LR_s4', (nviews, py_LR_s4, px_LR_s4, 3, 1),
                             chunks=(nviews, py_LR_s4, px_LR_s4, 3, 1),
                             maxshape=(nviews, py_LR_s4, px_LR_s4, 3, None))

dset_v_LR_s2 = file.create_dataset( 'stacks_v_LR_s2', (nviews, py_LR_s2, px_LR_s2, 3, 1),
                              chunks = (nviews, py_LR_s2, px_LR_s2, 3, 1),
                              maxshape = (nviews, py_LR_s2, px_LR_s2, 3, None))

dset_h_LR_s2 = file.create_dataset('stacks_h_LR_s2', (nviews, py_LR_s2, px_LR_s2, 3, 1),
                             chunks=(nviews, py_LR_s2, px_LR_s2, 3, 1),
                             maxshape=(nviews, py_LR_s2, px_LR_s2, 3, None))

dset_v_HR = file.create_dataset('stacks_v_HR', (nviews, py, px, 3, 1),
                                    chunks = (nviews, py, px, 3, 1),
                                    maxshape = (nviews, py, px, 3, None))

dset_h_HR = file.create_dataset('stacks_h_HR', (nviews, py, px, 3, 1),
                                   chunks=(nviews, py, px, 3, 1),
                                   maxshape=(nviews, py, px, 3, None))

# dataset for correcponsing center view patch (to train joint upsampling)
# ideally, would want to reconstruct full 4D LF patch, but probably too memory-intensive
# keep for future work
# dset_cv = file.create_dataset('cv', (py, px, 3, 1),
#                               chunks=(py, px, 3, 1),
#                               maxshape=(py, px, 3, None))

dset_disp_s2 = file.create_dataset('disp_s2', (py_LR_s2, px_LR_s2, 1),
                                   chunks=(py_LR_s2, px_LR_s2, 1),
                                   maxshape=(py_LR_s2, px_LR_s2, None))


dset_disp_HR = file.create_dataset('disp_s4', (py, px, 1),
                                   chunks=(py, px, 1),
                                   maxshape=(py, px, None))


#
# loop over all datasets, write out each dataset in patches
# to feed to autoencoder in random order
#
index = 0
idx_folder = 0
for lf_name in data_folders:

    data_folder = os.path.join(data_source, lf_name)
    print("now %i / %i" % (idx_folder+1, len(data_folders)))
    idx_folder = idx_folder+1
    # for i in range(0, 2):
    # if os.path.isdir(os.path.join(data_folder, inner_folder, flow[i])):
    # data_path = os.path.join(data_folder, inner_folder, flow[i])
    data_path = data_folder

    LF = file_io.read_lightfield_crosshair(data_path)
    # LF_LR = LF_LR.astype(np.float32)
    LF = LF.astype(np.float32)

    LF_2 = file_io.read_lightfield_crosshair(data_path)
    # LF_LR = LF_LR.astype(np.float32)
    LF_2 = LF_2.astype(np.float32)


    LF_temp = file_io.read_lightfield_crosshair(data_path)
    # LF_LR = LF_LR.astype(np.float32)
    LF_temp = LF_temp.astype(np.float32)

    LF_LR_s2 = np.zeros((LF.shape[0],LF.shape[1],int(LF.shape[2]/s2),
                    int(LF.shape[3]/s2),int(LF.shape[4])),np.float32)

    LF_LR_s4 = np.zeros((LF.shape[0], LF.shape[1], int(LF.shape[2] / s4),
                         int(LF.shape[3] / s4), int(LF.shape[4])), np.float32)
    #
    cv_gt = lf_tools.cv(LF_temp)

    ############################################################################################
    # evail hack: make the light field brighter
    ############################################################################################
    imean = 0.3
    factor = imean / np.mean(cv_gt)
    LF = LF * factor
    LF_2 = LF_2 * factor
    LF_temp = LF_temp * factor

    LF = np.clip(LF, 0, 1)
    LF_2 = np.clip(LF_2, 0, 1)
    LF_temp = np.clip(LF_temp, 0, 1)

    # LF_diffuse = LF_diffuse * factor
    # LF_specular = LF_specular * factor
    # LF = np.add(LF_diffuse, LF_specular)

    ############################################################################################
    ############################################################################################

    disp = file_io.read_disparity(data_path)
    disp_gt = np.array(disp[0])
    disp_gt = np.flip(disp_gt, 0)
    disp_gt_s2 = disp_gt[0:disp_gt.shape[0]-1:s2, 0:disp_gt.shape[1]-1:s2]
    # disp_LF = file_io.read_disparity_crosshair(data_path)
    # disp_44 = np.array(disp_LF[4,4,:,:])
    # disp44 = np.flip(disp_44,0)
    # err = sum(sum(disp_44-disp_gt))

    for v in range(0, nviews):
        for h in range(0, nviews):
            LF[v, h, :, :, :] = gaussian_filter(LF[v, h, :, :, :], sigma = 0.5, truncate=2)
            LF_LR_s2[v, h, :, :, :] = LF[v, h, 0:LF.shape[2]-1:s2, 0:LF.shape[3]-1:s2, :]

            LF_2[v, h, :, :, :] = gaussian_filter(LF_2[v, h, :, :, :], sigma=0.8, truncate=2)
            LF_LR_s4[v, h, :, :, :] = LF[v, h, 0:LF.shape[2] - 1:s4, 0:LF.shape[3] - 1:s4, :]


    # lf_tools.save_image(training_data_dir + 'input' + lf_name, cv_gt)

    # maybe we need those, probably not.
    # param_dict = file_io.read_parameters(data_folder)

    # write out one individual light field
    # block count
    cx_LR_s2 = np.int32((LF_LR_s2.shape[3] - px_LR_s2) / sx_LR_s2) + 1
    cy_LR_s2 = np.int32((LF_LR_s2.shape[2] - py_LR_s2) / sy_LR_s2) + 1
    cx_LR_s4 = np.int32((LF_LR_s4.shape[3] - px_LR_s4) / sx_LR_s4) + 1
    cy_LR_s4 = np.int32((LF_LR_s4.shape[2] - py_LR_s4) / sy_LR_s4) + 1
    cx_HR = np.int32((LF.shape[3] - px) / sx) + 1
    cy_HR = np.int32((LF.shape[2] - py) / sy) + 1


    for by in np.arange(0, cy_HR):
        sys.stdout.write('.')
        sys.stdout.flush()

        for bx in np.arange(0, cx_HR):
            x_LR_s2 = bx * sx_LR_s2
            y_LR_s2 = by * sx_LR_s2

            x_LR_s4 = bx * sx_LR_s4
            y_LR_s4 = by * sx_LR_s4

            x = bx * sx
            y = by * sx
            # extract data

            # scale 4
            (stack_h_LR_s4, stack_v_LR_s4) = lf_tools.epi_stacks(LF_LR_s4, y_LR_s4, x_LR_s4, py_LR_s4, px_LR_s4)
            # make sure the direction of the view shift is the first spatial dimension
            stack_h_LR_s4 = np.transpose(stack_h_LR_s4, (0, 2, 1, 3))

            # write to respective HDF5 datasets
            dset_v_LR_s4.resize(index + 1, 4)
            dset_v_LR_s4[:, :, :, :, index] = stack_v_LR_s4

            dset_h_LR_s4.resize(index + 1, 4)
            dset_h_LR_s4[:, :, :, :, index] = stack_h_LR_s4


            # scale 2
            (stack_h_LR_s2, stack_v_LR_s2) = lf_tools.epi_stacks(LF_LR_s2, y_LR_s2, x_LR_s2, py_LR_s2, px_LR_s2)
            # make sure the direction of the view shift is the first spatial dimension
            stack_h_LR_s2 = np.transpose(stack_h_LR_s2, (0, 2, 1, 3))

            # write to respective HDF5 datasets
            dset_v_LR_s2.resize(index + 1, 4)
            dset_v_LR_s2[:, :, :, :, index] = stack_v_LR_s2

            dset_h_LR_s2.resize(index + 1, 4)
            dset_h_LR_s2[:, :, :, :, index] = stack_h_LR_s2



            (stack_h_HR, stack_v_HR) = lf_tools.epi_stacks(LF_temp, y, x, py, px)
            # make sure the direction of the view shift is the first spatial dimension
            stack_h_HR = np.transpose(stack_h_HR, (0, 2, 1, 3))

            # write to respective HDF5 datasets
            dset_v_HR.resize(index + 1, 4)
            dset_v_HR[:, :, :, :, index] = stack_v_HR

            dset_h_HR.resize(index + 1, 4)
            dset_h_HR[:, :, :, :, index] = stack_h_HR


            disp_HR = disp_gt[y:y + py, x:x + px]
            disp_s2 = disp_gt_s2[y_LR_s2:y_LR_s2 + py_LR_s2, x_LR_s2:x_LR_s2 + px_LR_s2]

            dset_disp_HR.resize(index + 1, 2)
            dset_disp_HR[:, :, index] = disp_HR

            dset_disp_s2.resize(index + 1, 2)
            dset_disp_s2[:, :, index] = disp_s2

            # next patch
            index = index + 1

# next dataset
print(' done.')

