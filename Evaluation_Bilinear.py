import file_io
import matplotlib.pyplot as plt
import h5py
from scipy.misc import imresize
from skimage import measure
import os
import lf_tools
import numpy as np
import scipy.io as scio
from scipy.misc import imsave

data_folders = []
# data_folders.append('/home/mz/HD_data/SR_data_backups/TEST/BENCHMARK test/')
# data_folders.append('/home/mz/HD_data/SR_data_backups/TEST/HCI test/')
# data_folders.append('/home/mz/HD_data/SR_data_backups/TEST/LYTRO test/')
# data_folders.append('/home/mz/HD_data/SR_data_backups/TEST/NEW test/')
# data_folders.append('/home/mz/HD_data/SR_data_backups/TEST/STANFORD test/')
# data_folders.append('/home/mz/HD_data/SR_data_backups/TEST/GraphBased/')
data_folders.append('/home/mz/HD_data/SR_data_backups/TEST/final/')
# dest_path = '/home/mz/HD_data/SR_data_backups/TEST/Graph_Based_TEST/'
dest_path = '/home/mz/HD_data/SR_data_backups/TEST/'


for i in range(0,len(data_folders)):
    data_folder = os.listdir(data_folders[i])
    for lf_names in data_folder:
        data = data_folders[i]+lf_names
        # lf_name = lf_names[8:-5]
        lf_name = lf_names[:-5]
        f = h5py.File(data, 'r')
        LF = f['LF']
        LF_temp = f['LF']
        cv_gt = lf_tools.cv(LF)
        H = cv_gt.shape[0]
        W = cv_gt.shape[1]
        C = cv_gt.shape[-1]

        print(lf_name)
        print(H)
        print(W)


        cv_s2 = cv_gt[0:H - 1:2, 0:W - 1:2, :]
        cv_s4 = cv_gt[0:H - 1:4, 0:W - 1:4, :]



        # if H == 434:
        #     H = H-2
        #     W = W-5
        #
        #     LF_2 = np.zeros([LF.shape[0],LF.shape[1],int(H/2),int(W/2),C])
        #     LF_4 = np.zeros([LF.shape[0], LF.shape[1], int(H / 4), int(W / 4), C])
        #     LF_HR = np.zeros([LF.shape[0], LF.shape[1], H, W, C])
        #     for v in range(0,LF.shape[0]):
        #         for h in range(0,LF.shape[1]):
        #             LF_2[v,h,:,:,:] = imresize(LF_temp[v,h,0:-2,0:-5,:], [int(H/2), int(W/2)], 'bicubic')/255
        #             LF_4[v, h, :, :, :] = imresize(LF_temp[v, h, 0:-2,0:-5, :], [int(H / 4), int(W / 4)], 'bicubic') / 255
        #             LF_HR[v,h,:,:,:] = LF[v,h,0:-2,0:-5,:]
        #
        # elif H == 376:
        #     H = H
        #     W = W-1
        #
        #     LF_2 = np.zeros([LF.shape[0], LF.shape[1], int(H / 2), int(W / 2), C])
        #     LF_4 = np.zeros([LF.shape[0], LF.shape[1], int(H / 4), int(W / 4), C])
        #     LF_HR = np.zeros([LF.shape[0], LF.shape[1], H, W, C])
        #     for v in range(0, LF.shape[0]):
        #         for h in range(0, LF.shape[1]):
        #             LF_2[v, h, :, :, :] = imresize(LF_temp[v, h, :, 0:-1, :], [int(H / 2), int(W / 2)],
        #                                            'bicubic') / 255
        #             LF_4[v, h, :, :, :] = imresize(LF_temp[v, h, :, 0:-1, :], [int(H / 4), int(W / 4)],
        #                                            'bicubic') / 255
        #             LF_HR[v, h, :, :, :] = LF[v, h, :, 0:-1, :]
        #
        # elif H == 926:
        #     H = H -2
        #     W = W - 2
        #
        #     LF_2 = np.zeros([LF.shape[0], LF.shape[1], int(H / 2), int(W / 2), C])
        #     LF_4 = np.zeros([LF.shape[0], LF.shape[1], int(H / 4), int(W / 4), C])
        #     LF_HR = np.zeros([LF.shape[0], LF.shape[1], H, W, C])
        #     for v in range(0, LF.shape[0]):
        #         for h in range(0, LF.shape[1]):
        #             LF_2[v, h, :, :, :] = imresize(LF_temp[v, h, 1:-1, 1:-1, :], [int(H / 2), int(W / 2)],
        #                                            'bicubic') / 255
        #             LF_4[v, h, :, :, :] = imresize(LF_temp[v, h, 1:-1, 1:-1, :], [int(H / 4), int(W / 4)],
        #                                            'bicubic') / 255
        #             LF_HR[v, h, :, :, :] = LF[v, h, 1:-1, 1:-1, :]
        #
        #
        #
        # mat_name_HR = lf_name + '_HR.mat'
        # mat_name_LR = lf_name + '_LR.mat'
        # mat_name_LR_4 = lf_name + '_LR_4.mat'
        # scio.savemat(dest_path + mat_name_HR, {'LF_HR':LF_HR})
        # scio.savemat(dest_path + mat_name_LR, {'LF_LR':LF_2})
        # scio.savemat(dest_path + mat_name_LR_4, {'LF_LR_s4': LF_4})
        # k = 0

        res_s2_L = imresize(cv_s2, [H, W], 'bilinear')
        # res_s2_C = imresize(cv_s2, [H, W], 'bicubic')/255
        res_s4_L = imresize(cv_s4, [H, W], 'bilinear')
        # res_s4_C = imresize(cv_s4, [H, W], 'bicubic')/255
        imsave(dest_path+lf_name+'_s2.png',res_s2_L)
        imsave(dest_path + lf_name + '_s4.png', res_s4_L)


        # PSNR_2_L = measure.compare_psnr(cv_gt, res_s2_L, data_range=1, dynamic_range=None)
        # SSIM_2_L = measure.compare_ssim(cv_gt, res_s2_L, data_range=1, multichannel=True)
        #
        # PSNR_2_C = measure.compare_psnr(cv_gt, res_s2_C, data_range=1, dynamic_range=None)
        # SSIM_2_C = measure.compare_ssim(cv_gt, res_s2_C, data_range=1, multichannel=True)
        #
        # PSNR_4_L = measure.compare_psnr(cv_gt, res_s4_L, data_range=1, dynamic_range=None)
        # SSIM_4_L = measure.compare_ssim(cv_gt, res_s4_L, data_range=1, multichannel=True)
        #
        # PSNR_4_C = measure.compare_psnr(cv_gt, res_s4_C, data_range=1, dynamic_range=None)
        # SSIM_4_C = measure.compare_ssim(cv_gt, res_s4_C, data_range=1, multichannel=True)
        #
        # print(lf_name)
        # print('PSNR S2: ', PSNR_2_L)
        # print('SSIM S2: ', SSIM_2_L)
        # print('PSNR S4: ', PSNR_4_L)
        # print('SSIM S4: ', SSIM_4_L)

        # print(lf_name)
        # print('PSNR S2: ', PSNR_2_C)
        # print('SSIM S2: ', SSIM_2_C)
        # print('PSNR S4: ', PSNR_4_C)
        # print('SSIM S4: ', SSIM_4_C)








