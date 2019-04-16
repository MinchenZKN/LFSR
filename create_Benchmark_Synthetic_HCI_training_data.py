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

training_data_dir = "/home/mz/HD_data/CVPR_Sup_Mat/light_field_512/"
training_data_filename = 'lf_patch_try.hdf5'
check_path = "/home/mz/HD_data/CVPR_Sup_Mat/light_field_512/img_check_3/"

data_source = "/home/mz/HD_data/CVPR_data_used/light_field_512/"


file = h5py.File( training_data_dir + training_data_filename, 'w' )

data_folders_all = os.listdir(data_source)

# EPI patches, nviews x patch size x patch size x channels
# horizontal and vertical direction (to get crosshair)

dset_v_HR = file.create_dataset('stacks_v_HR', (nviews, py, px, 3, 1),
                                    chunks = (nviews, py, px, 3, 1),
                                    maxshape = (nviews, py, px, 3, None))

dset_h_HR = file.create_dataset('stacks_h_HR', (nviews, py, px, 3, 1),
                                   chunks=(nviews, py, px, 3, 1),
                                   maxshape=(nviews, py, px, 3, None))

index = 0
idx_folder = 0
for folder in data_folders_all:
    if folder == 'Benchmark':
        data_folders = os.listdir(data_source+folder+'/')
        for lf_name in data_folders:
            data_folder = os.path.join(data_source, lf_name)
            print("now %i / %i" % (idx_folder+1, len(data_folders)))
            idx_folder = idx_folder+1

            data_path = data_source+folder+'/'+lf_name
            LF_temp = file_io.read_lightfield(data_path)
            LF_temp = LF_temp.astype(np.float32)

            imwrite(check_path + lf_name + '_v1.png', LF_temp[0, 4, :, :, :])
            imwrite(check_path + lf_name + '_v2.png', LF_temp[8, 4, :, :, :])
            imwrite(check_path + lf_name + '_h1.png', LF_temp[4, 0, :, :, :])
            imwrite(check_path + lf_name + '_h2.png', LF_temp[4, 8, :, :, :])

            # write out one individual light field
            # block count
            cx_HR = np.int32((LF_temp.shape[3] - px) / sx) + 1
            cy_HR = np.int32((LF_temp.shape[2] - py) / sy) + 1

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

                    (stack_h_HR, stack_v_HR) = lf_tools.epi_stacks(LF_temp, y, x, py, px)
                    # make sure the direction of the view shift is the first spatial dimension
                    stack_h_HR = np.transpose(stack_h_HR, (0, 2, 1, 3))

                    # write to respective HDF5 datasets
                    dset_v_HR.resize(index + 1, 4)
                    dset_v_HR[:, :, :, :, index] = stack_v_HR

                    dset_h_HR.resize(index + 1, 4)
                    dset_h_HR[:, :, :, :, index] = stack_h_HR

                    # next patch
                    index = index + 1

    if folder == 'cross_light_field_512':
        clip_max = 2
        data_folders = sorted(os.listdir(data_source+folder+'/'))
        for i in range(0,len(data_folders)):
            all_folders = os.listdir(data_source+folder + '/'+ data_folders[i]+'/')
            for lf_name in all_folders:
                data_folder = data_source+folder + '/'+ data_folders[i]+'/' + lf_name + '/sequence/000000/'
                print("now %i / %i" % (idx_folder+1, len(data_folders)))
                idx_folder = idx_folder+1

                # read diffuse color
                LF_dc = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'dc')
                # read diffuse direct
                LF_dd = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'dd')
                # read diffuse indirect
                LF_di = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'di')
                # read glossy color
                LF_gc = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'gc')
                # read glossy direct
                LF_gd = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'gd')
                # read glossy indirect
                LF_gi = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'gi')

                # albedo LF
                LF_albedo = LF_dc
                # shading LF
                LF_sh = np.add(LF_dd, LF_di)

                min_v = np.amin(LF_sh)
                max_v = np.amax(LF_sh)
                if max_v > 2:
                    print('rescaling')
                    LF_sh_old = LF_sh
                    LF_sh = np.multiply(np.divide(LF_sh, max_v), clip_max)

                    # find scale constant
                    tmp = LF_sh_old
                    tmp[LF_sh_old == 0] = 1
                    alpha = np.divide(LF_sh, tmp)
                    alpha[LF_sh_old == 0] = 1
                    alpha[np.isnan(alpha)] = 1
                    alpha[np.isinf(alpha)] = 1
                    del LF_sh_old
                else:
                    alpha = 1

                # glossy LF
                LF_specular = np.multiply(LF_gc, np.add(LF_gd, LF_gi))
                LF_specular = np.multiply(alpha, LF_specular)
                # diffuse LF
                LF_diffuse = np.multiply(LF_albedo, LF_sh)
                # show center view
                cv_diffuse = lf_tools.cv(LF_diffuse)
                # show center view
                cv_specular = lf_tools.cv(LF_specular)
                # lf_tools.save_image( training_data_dir + 'specular' +lf_name, cv_specular)
                # input LF
                LF = np.add(LF_diffuse, LF_specular)
                cv_gt = lf_tools.cv(LF)

                imwrite(check_path + lf_name + '_v1.png', LF[0, 4, :, :, :])
                imwrite(check_path + lf_name + '_v2.png', LF[8, 4, :, :, :])
                imwrite(check_path + lf_name + '_h1.png', LF[4, 0, :, :, :])
                imwrite(check_path + lf_name + '_h2.png', LF[4, 8, :, :, :])

                # write out one individual light field
                # block count
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

                        (stack_h_HR, stack_v_HR) = lf_tools.epi_stacks(LF, y, x, py, px)
                        # make sure the direction of the view shift is the first spatial dimension
                        stack_h_HR = np.transpose(stack_h_HR, (0, 2, 1, 3))

                        # # write to respective HDF5 datasets
                        dset_v_HR.resize(index + 1, 4)
                        dset_v_HR[:, :, :, :, index] = stack_v_HR

                        dset_h_HR.resize(index + 1, 4)
                        dset_h_HR[:, :, :, :, index] = stack_h_HR

                        # next patch
                        index = index + 1


    # elif folder == 'HCI':
    #     data_folders = sorted(os.listdir(data_source+folder+'/'))
    #     for lf_name in data_folders:
    #         data_folder = data_source + folder + '/' + lf_name + '/lf.h5'
    #         print("now %i / %i" % (idx_folder+1, len(data_folders)))
    #         idx_folder = idx_folder+1
    #
    #         data_path = data_folder
    #         f = h5py.File(data_path, 'r')
    #         LF_temp = f['LF'].value
    #         LF_temp = LF_temp.astype(np.float32) / 255.0
    #         LF_temp = np.flip(LF_temp, axis=1)
    #
    #         imwrite(check_path + lf_name + '_v1.png', LF_temp[0, 4, :, :, :])
    #         imwrite(check_path + lf_name + '_v2.png', LF_temp[8, 4, :, :, :])
    #         imwrite(check_path + lf_name + '_h1.png', LF_temp[4, 0, :, :, :])
    #         imwrite(check_path + lf_name + '_h2.png', LF_temp[4, 8, :, :, :])
    #
    #         # write out one individual light field
    #         # block count
    #         cx_HR = np.int32((LF_temp.shape[3] - px) / sx) + 1
    #         cy_HR = np.int32((LF_temp.shape[2] - py) / sy) + 1
    #
    #         for by in np.arange(0, cy_HR):
    #             sys.stdout.write('.')
    #             sys.stdout.flush()
    #
    #             for bx in np.arange(0, cx_HR):
    #                 x_LR_s2 = bx * sx_LR_s2
    #                 y_LR_s2 = by * sx_LR_s2
    #
    #                 x_LR_s4 = bx * sx_LR_s4
    #                 y_LR_s4 = by * sx_LR_s4
    #
    #                 x = bx * sx
    #                 y = by * sx
    #                 # extract data
    #
    #                 (stack_h_HR, stack_v_HR) = lf_tools.epi_stacks(LF_temp, y, x, py, px)
    #                 # make sure the direction of the view shift is the first spatial dimension
    #                 stack_h_HR = np.transpose(stack_h_HR, (0, 2, 1, 3))
    #
    #                 # write to respective HDF5 datasets
    #                 dset_v_HR.resize(index + 1, 4)
    #                 dset_v_HR[:, :, :, :, index] = stack_v_HR
    #
    #                 dset_h_HR.resize(index + 1, 4)
    #                 dset_h_HR[:, :, :, :, index] = stack_h_HR
    #
    #                 # next patch
    #                 index = index + 1

# next dataset
print(' done.')