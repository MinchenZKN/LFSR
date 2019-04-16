import numpy as np
import h5py
import os
import sys
import file_io

import lf_tools
import matplotlib.pyplot as plt
import scipy


def refocus_cross(cv, disp, nviews):

    nt=nviews
    ns=nviews


    if cv.ndim == 3:
        height = cv.shape[0]
        width = cv.shape[1]
        channels = cv.shape[2]
        cv_v = np.tile(cv, (nt, 1, 1, 1))
        cv_h = np.tile(cv.transpose(1, 0, 2), (ns, 1, 1, 1))
    else:
        height = cv.shape[1]
        width = cv.shape[2]
        channels = cv.shape[3]
        cv_v = np.tile(cv, (nt,1, 1, 1, 1))
        cv_v = cv_v.transpose(1, 0, 2, 3, 4)
        cv_h = np.tile(cv.transpose(0, 2, 1, 3), (ns, 1, 1, 1, 1))
        cv_h = cv_h.transpose(1, 0, 2, 3, 4)



    if disp.ndim == 2:
        disp_v = np.tile(disp,(nt,1,1))
        disp_h = np.tile(disp.transpose(1,0),(ns,1,1))
    else:
        disp_v = np.tile(disp, (nt, 1, 1, 1))
        disp_h = np.tile(disp.transpose(0, 2, 1), (ns, 1, 1, 1))
        disp_v = disp_v.transpose(1, 0, 2, 3)
        disp_h = disp_h.transpose(1, 0, 2, 3)
        batch = disp.shape[0]



    disp_v = np.nan_to_num(disp_v)
    disp_h = np.nan_to_num(disp_h)
    c = (nt + 1) / 2 - 1

    [p, q, v, u] = np.meshgrid(np.arange(batch), np.arange(nt), np.arange(height), np.arange(width))
    pp = p.transpose(1, 0, 2, 3)
    qq = q.transpose(1, 0, 2, 3)
    vv = v.transpose(1, 0, 2, 3)
    uu = u.transpose(1, 0, 2, 3)

    y = (qq - c) * (disp_v) + vv
    x = (qq - c) * (disp_h) + vv

    points_v = [np.arange(batch), np.arange(nt), np.arange(height), np.arange(width)]
    points_h = [np.arange(batch), np.arange(ns), np.arange(height), np.arange(width)]
    a = np.clip(y.ravel(), 0, height-1)
    b = np.clip(x.ravel(), 0, width-1)
    xi_v = (pp.ravel(), qq.ravel(), a, uu.ravel())
    xi_h = (pp.ravel(), qq.ravel(), b, uu.ravel())

    stack_v = np.zeros([batch, nt, height, width, channels])
    stack_h = np.zeros([batch, ns, height, width, channels])


    for c in range(channels):
        stack_v[:, :, :, :, c] = scipy.interpolate.interpn(points=points_v, values=cv_v[:,:,:,:,c], xi=xi_v).reshape((batch, nt, height, width))
        stack_h[:, :, :, :, c] = scipy.interpolate.interpn(points=points_h, values=cv_h[:,:,:,:,c], xi=xi_h).reshape((batch, nt, height, width))
        # stack_v[:, :, :, :, c] = scipy.interpolate.RegularGridInterpolator(points=points_v, values=cv_v[:, :, :, :, c])
        # stack_h[:, :, :, :, c] = scipy.interpolate.RegularGridInterpolator(points=points_h, values=cv_h[:, :, :, :, c])

    return stack_h, stack_v


data_source = "/home/mz/HD data/SR data backups/full_data_512/"
# data_folders = os.listdir(data_source)
data_folders = []
# data_folders.append('dishes')
data_folders.append('greek')
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

cv = np.zeros([2,512,512,3])
dispgt = np.zeros([2,512,512])
n = 0
for lf_name in data_folders:

    data_folder = os.path.join(data_source, lf_name)

    LF = file_io.read_lightfield(data_folder)
    LF = LF.astype(np.float32)

    disp = file_io.read_disparity(data_folder)
    disp_gt = np.array(disp[0])
    disp_gt = np.flip(disp_gt, 0)
    dispgt[n, :,:] = disp_gt

    cv[n,:,:,:] = lf_tools.cv(LF)
    n=n+1

stack_h, stack_v = refocus_cross(cv,dispgt,9)

    # for k in range(0,9):
    #     plt.figure(k)
    #     plt.imshow(stack_h[k, :, :, :])
    #     plt.show()
    #

    # for k in range(0, 9):
    #     plt.figure(k)
    #     plt.imshow(np.abs(stack_h[k, :, :, :] - cv.transpose(1,0,2)))
k=0

########################################################################################################################################
# training_data_dir = "/home/mz/HD data/"
# training_data_filename = 'tf_test.hdf5'
# file = h5py.File(training_data_dir + training_data_filename, 'w')
#
# data_source = "/home/mz/HD data/SR data backups/full_data_512/"
# # data_folders = os.listdir(data_source)
# data_folders = []
# data_folders.append('dishes')
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
#
# nviews = 9
# py_LR = 512
# px_LR = 512
# py = 512
# px = 512
#
#
# dset_v = file.create_dataset( 'stacks_v', (nviews, py_LR, px_LR, 3, 1),
#                               chunks = (nviews, py_LR, px_LR, 3, 1),
#                               maxshape = (nviews, py_LR, px_LR, 3, None))
#
# dset_h = file.create_dataset('stacks_h', (nviews, py_LR, px_LR, 3, 1),
#                              chunks=(nviews, py_LR, px_LR, 3, 1),
#                              maxshape=(nviews, py_LR, px_LR, 3, None))
#
#
# # dataset for correcponsing center view patch (to train joint upsampling)
# # ideally, would want to reconstruct full 4D LF patch, but probably too memory-intensive
# # keep for future work
# dset_cv = file.create_dataset('cv', (py, px, 3, 1),
#                               chunks=(py, px, 3, 1),
#                               maxshape=(py, px, 3, None))
#
# dset_disp = file.create_dataset('disp', (py, px, 1),
#                                 chunks=(py, px, 1),
#                                 maxshape=(py, px, None))
# n = 0
# for lf_name in data_folders:
#
#     data_folder = os.path.join(data_source, lf_name)
#
#     LF = file_io.read_lightfield(data_folder)
#     LF = LF.astype(np.float32)
#
#     disp = file_io.read_disparity(data_folder)
#     disp_gt = np.array(disp[0])
#     disp_gt = np.flip(disp_gt, 0)
#     dset_disp.resize(n + 1, 2)
#     dset_disp[:, :, n] = disp_gt
#
#     cv = lf_tools.cv(LF)
#     dset_cv.resize(n + 1, 3)
#     dset_cv[:, :, :, n] = cv
#
#     (stack_h_gt, stack_v_gt) = lf_tools.epi_stacks(LF, 0, 0, 512, 512)
#     stack_h_gt = np.transpose(stack_h_gt, (0, 2, 1, 3))
#     dset_v.resize(n + 1, 4)
#     dset_v[:, :, :, :, n] = stack_v_gt
#
#     dset_h.resize(n + 1, 4)
#     dset_h[:, :, :, :, n] = stack_h_gt
#     n=n+1
#
