import numpy as np
import scipy.io as sc
import h5py
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

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
from imageio import imwrite

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

# training_data_dir = "/home/mz/HD_data/CVPR_Sup_Mat/Lytro/"
# training_data_filename = 'lf_patch_lytro_2.hdf5'
# check_path = "/home/mz/HD_data/CVPR_Sup_Mat/Lytro/img_check/"

data_source = "/home/mz/HD_data/CVPR_data_used/Lytro_using/"
name_ending = '_lf.mat'

# file = h5py.File( training_data_dir + training_data_filename, 'w' )

data_folders_all = os.listdir(data_source)

# EPI patches, nviews x patch size x patch size x channels
# horizontal and vertical direction (to get crosshair)

# dset_v_HR = file.create_dataset('stacks_v_HR', (nviews, py, px, 3, 1),
#                                     chunks = (nviews, py, px, 3, 1),
#                                     maxshape = (nviews, py, px, 3, None))
#
# dset_h_HR = file.create_dataset('stacks_h_HR', (nviews, py, px, 3, 1),
#                                    chunks=(nviews, py, px, 3, 1),
#                                    maxshape=(nviews, py, px, 3, None))

#
index = 0
idx_folder = 0
for folder in data_folders_all:
    if folder == 'Flowers':
        data_folders = os.listdir(data_source+folder+'/')
        for lf_name in data_folders:
            real_name = lf_name[0:-8]
            data_folder = os.path.join(data_source, lf_name)
            print("now %i / %i" % (idx_folder+1, len(data_folders)))
            idx_folder = idx_folder+1

            data_path = data_source+folder+'/'+lf_name
            f = h5py.File(data_path, 'r')


            # # flowers and sythetic
            LF_temp = np.transpose(f['LF'], (4, 3, 2, 1, 0))
            LF_temp = LF_temp.astype(np.float32)

            cv_gt = lf_tools.cv(LF_temp)

            # ############################################################################################
            # # evail hack: make the light field brighter
            # ############################################################################################
            if lf_name[0:3] == 'IMG':
                imean = 0.15
                factor = imean / np.mean(cv_gt)

                LF_temp = LF_temp * factor

                LF_temp = np.clip(LF_temp, 0, 1)

            # ############################################################################################
            # ############################################################################################

            # imwrite(check_path + real_name + '_v1.png', LF_temp[0, 4, :, :, :])
            # imwrite(check_path + real_name + '_v2.png', LF_temp[8, 4, :, :, :])
            # imwrite(check_path + real_name + '_h1.png', LF_temp[4, 0, :, :, :])
            # imwrite(check_path + real_name + '_h2.png', LF_temp[4, 8, :, :, :])

            # write out one individual light field
            # block count
            cx_HR = np.int32((LF_temp.shape[3] - px) / sx) + 1
            cy_HR = np.int32((LF_temp.shape[2] - py) / sy) + 1

            for by in np.arange(0, cy_HR):
                sys.stdout.write('.')
                sys.stdout.flush()

                for bx in np.arange(0, cx_HR):

                    x = bx * sx
                    y = by * sx
                    # extract data

                    (stack_h_HR, stack_v_HR) = lf_tools.epi_stacks(LF_temp, y, x, py, px)
                    # make sure the direction of the view shift is the first spatial dimension
                    stack_h_HR = np.transpose(stack_h_HR, (0, 2, 1, 3))

                    # write to respective HDF5 datasets
                    # dset_v_HR.resize(index + 1, 4)
                    # dset_v_HR[:, :, :, :, index] = stack_v_HR
                    #
                    # dset_h_HR.resize(index + 1, 4)
                    # dset_h_HR[:, :, :, :, index] = stack_h_HR

                    # next patch
                    index = index + 1

    elif folder == 'K_H_O':
        data_folders = os.listdir(data_source+folder+'/')
        for lf_name in data_folders:
            real_name = lf_name[0:-8]
            data_folder = os.path.join(data_source, lf_name)
            print("now %i / %i" % (idx_folder+1, len(data_folders)))
            idx_folder = idx_folder+1

            data_path = data_path = data_source+folder+'/'+lf_name
            f = h5py.File(data_path, 'r')

            # # K_H and Owl
            LF_temp = np.transpose(f['LF'], (4, 3, 2, 1, 0))[3:12,3:12,:,:,0:3] # K_H
            LF_temp = (LF_temp/65535).astype(np.float32)

            # ############################################################################################
            # ############################################################################################
            # imwrite(check_path + real_name + '_v1.png', LF_temp[0, 4, :, :, :])
            # imwrite(check_path + real_name + '_v2.png', LF_temp[8, 4, :, :, :])
            # imwrite(check_path + real_name + '_h1.png', LF_temp[4, 0, :, :, :])
            # imwrite(check_path + real_name + '_h2.png', LF_temp[4, 8, :, :, :])
            # write out one individual light field
            # block count
            cx_HR = np.int32((LF_temp.shape[3] - px) / sx) + 1
            cy_HR = np.int32((LF_temp.shape[2] - py) / sy) + 1
            #
            for by in np.arange(0, cy_HR):
                sys.stdout.write('.')
                sys.stdout.flush()

                for bx in np.arange(0, cx_HR):

                    x = bx * sx
                    y = by * sx
                    # extract data

                    (stack_h_HR, stack_v_HR) = lf_tools.epi_stacks(LF_temp, y, x, py, px)
                    # make sure the direction of the view shift is the first spatial dimension
                    stack_h_HR = np.transpose(stack_h_HR, (0, 2, 1, 3))

                    # write to respective HDF5 datasets
                    # dset_v_HR.resize(index + 1, 4)
                    # dset_v_HR[:, :, :, :, index] = stack_v_HR
                    #
                    # dset_h_HR.resize(index + 1, 4)
                    # dset_h_HR[:, :, :, :, index] = stack_h_HR

                    # next patch
                    index = index + 1

# next dataset
print(' done.')

