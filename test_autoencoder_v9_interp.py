
from queue import Queue
import time
import numpy as np
import h5py
# plotting
import matplotlib.pyplot as plt
from scipy.misc import imresize
import cv2 as cv

from skimage import measure
from skimage.color import lab2rgb, yuv2rgb
from libs.convert_colorspace import rgb2YCbCr, YCbCr2rgb, rgb2YUV
from interp import *
# timing and multithreading
import _thread

# light field GPU tools
import lf_tools
# evaluator thread
from encode_decode_lightfield_v9_interp import encode_decode_lightfield
from encode_decode_lightfield_v9_interp import scale_back
from thread_evaluate_v9 import evaluator_thread

# configuration
import config_autoencoder_ycbcr_3d as hp


# Model path setup
ckpt_number = 200
model_id = hp.network_model
model_path = './networks/' + model_id + '/model_'+ str(ckpt_number) +'.ckpt'
result_folder = hp.eval_res['result_folder']
data_eval_folder = hp.eval_res['test_data_folder']

# I/O queues for multithreading
inputs = Queue( 15*15 )
outputs = Queue( 15*15 )

data_folders = (

( "super_resolution", "benchmark", "not seen", "lf_test_benchmark_antinous", "s4" ),
# ( "super_resolution", "benchmark", "not seen", "lf_test_benchmark_antinous", "s4" ),
# ( "super_resolution", "benchmark", "not seen", "lf_test_benchmark_herbs", "s2"  ),
# ( "super_resolution", "benchmark", "not seen", "lf_test_benchmark_cotton", "s2"  ),

# ( "super_resolution", "stanford", "not seen", "lf_test_stanford_HR_Bunny", "s4" ),
# ( "super_resolution", "stanford", "not seen", "lf_test_stanford_HR_Eucalyptus", "s4" ),
( "super_resolution", "stanford", "not seen", "lf_test_stanford_HR_JellyBeans", "s4" ),
# ( "super_resolution", "stanford", "not seen", "lf_test_stanford_HR_Treasure", "s4" ),
( "super_resolution", "stanford", "not seen", "lf_test_stanford_HR_Truck", "s4" ),

# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_pyramide", "s2" ),
# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_statue", "s2" ),
# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_couple", "s2" ),
# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_cube", "s2" ),
# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_buddha2", "s2"),
( "super_resolution", "HCI", "not seen", "lf_test_HCI_horses", "s2" ),
# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_medieval", "s2" ),
# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_monasRoom", "s2" ),
# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_papillon", "s2" ),
# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_stillLife", "s2" ),

( "super_resolution", "lytro", "not seen", "lf_test_lytro_flowers", "s4" ),
# ( "super_resolution", "lytro", "not seen", "lf_test_lytro_hedgehog3", "s2" ),
# ( "super_resolution", "lytro", "not seen", "lf_test_lytro_koala", "s2" ),
# ( "super_resolution", "lytro", "not seen", "lf_test_lytro_origami", "s2" ),
( "super_resolution", "lytro", "not seen", "lf_test_lytro_owl_str", "s2" ),
# ( "super_resolution", "lytro", "not seen", "lf_test_lytro_owl2", "s2" ),

# ( "super_resolution", "synthetic", "not seen", "lf_test_synthetic_new_1", "s4" ),
# ( "super_resolution", "synthetic", "not seen", "lf_test_synthetic_new_2", "s4" ),
# ( "super_resolution", "synthetic", "not seen", "lf_test_synthetic_new_3", "s4" ),
# ( "super_resolution", "synthetic", "not seen", "lf_test_synthetic_new_4", "s4" ),
# ( "super_resolution", "synthetic", "not seen", "lf_test_synthetic_new_5", "s4" ),
# ( "super_resolution", "synthetic", "not seen", "lf_test_synthetic_new_6", "s4" ),
# ( "super_resolution", "synthetic", "not seen", "lf_test_synthetic_new_7", "s4" ),
# ( "super_resolution", "synthetic", "not seen", "lf_test_synthetic_new_8", "s4" ),
# ( "super_resolution", "synthetic", "not seen", "lf_test_synthetic_new_9", "s4" ),
# ( "super_resolution", "synthetic", "not seen", "lf_test_synthetic_new_10", "s4" ),

)

# YCBCR
# evaluator thread
scales = ['s2', 's4'] # ,'s4'
_thread.start_new_thread( evaluator_thread,
                          ( model_path, hp, inputs,  outputs, scales ))

# wait a bit to not skew timing results with initialization
time.sleep(20)

# loop over all datasets and collect errors
results = []


for lf_name in data_folders:
    file = h5py.File(result_folder + lf_name[3] + '.hdf5', 'w')
    data_file = data_eval_folder + lf_name[0] + "/" + lf_name[1] + "/" + lf_name[2] + "/" + \
                    lf_name[3] + ".hdf5"
    hdf_file = h5py.File(data_file, 'r')
    LF_HR = np.clip(hdf_file['LF'],0,1)
    LF_HR_orig = LF_HR
    sh_orig = LF_HR_orig.shape
    if lf_name[-1] == 's2':
        sh_orig_s2 = np.asarray(sh_orig)
        sh_orig_s4 = np.zeros(sh_orig_s2.shape, np.int32)
        sh_orig_s4[0] = sh_orig_s2[0]
        sh_orig_s4[1] = sh_orig_s2[1]
        sh_orig_s4[2] = np.int(2 * sh_orig_s2[2])
        sh_orig_s4[3] = np.int(2 * sh_orig_s2[3])
        sh_orig_s4[4] = sh_orig_s2[4]
    else:
        sh_orig_s4 = np.asarray(sh_orig)
        sh_orig_s2 = np.zeros(sh_orig_s4.shape, np.int32)
        sh_orig_s2[0] = sh_orig_s4[0]
        sh_orig_s2[1] = sh_orig_s4[1]
        sh_orig_s2[2] = np.int(0.5*sh_orig_s4[2])
        sh_orig_s2[3] = np.int(0.5 * sh_orig_s4[3])
        sh_orig_s2[4] = sh_orig_s4[4]


    # process shapes
    if sh_orig[2] % 192 !=0:
        y_pad = 192 - sh_orig[2] % 192
    else:
        y_pad = 0

    if sh_orig[3] % 192 !=0:
        x_pad = 192 - sh_orig[3] % 192
    else:
        x_pad = 0

    LF_HR = np.pad(LF_HR, ((0,0),(0,0),(0,y_pad),(0,x_pad),(0,0)), mode = 'constant')
    sh = LF_HR.shape

    LF_gt = dict()
    LF_crosshair = dict()
    cv_gt = dict()
    LF_gt['input_'+lf_name[-1]] = LF_HR
    LF_crosshair['input_'+lf_name[-1]] = np.zeros((2,sh[1],sh[2],sh[3],sh[4]), np.float32)
    LF_crosshair['input_'+lf_name[-1]][0,:,:,:,:] = LF_HR[:,4,:,:,:]
    LF_crosshair['input_'+lf_name[-1]][1, :, :, :, :] = LF_HR[4, :, :, :, :]
    cv_gt['input_'+lf_name[-1]] = LF_HR[4,4,:,:,:]

    scale2 = np.zeros((sh[0], sh[1], np.int(0.5 * sh[2]), np.int(0.5 * sh[3]), sh[4]), np.float32)
    scale2[:, 4, :, :, :] = np.stack([cv.resize(LF_HR[i, 4, :, :, :], (np.int(0.5 * sh[3]), np.int(0.5 * sh[2])),
                                                interpolation=cv.INTER_CUBIC) for i in range(0, sh[0])])
    scale2[4, :, :, :, :] = np.stack([cv.resize(LF_HR[4, i, :, :, :], (np.int(0.5 * sh[3]), np.int(0.5 * sh[2])),
                                                interpolation=cv.INTER_CUBIC) for i in range(0, sh[0])])
    scale2_crosshair = np.zeros((2, sh[1], np.int(0.5 * sh[2]), np.int(0.5 * sh[3]), sh[4]), np.float32)
    scale2_crosshair[0, :, :, :, :] = scale2[:, 4, :, :, :]
    scale2_crosshair[1, :, :, :, :] = scale2[4, :, :, :, :]
    if lf_name[-1] == "s2":
        LF_gt['input'] = np.clip(scale2,0,1)
        LF_crosshair['input'] = np.clip(scale2_crosshair,0,1)
        cv_gt['input'] = np.clip(scale2[4,4,:,:,:],0,1)
    else:
        LF_gt['input_'+'s2'] = np.clip(scale2,0,1)
        LF_crosshair['input_'+'s2'] = np.clip(scale2_crosshair,0,1)
        cv_gt['input_'+'s2'] = np.clip(scale2[4,4,:,:,:],0,1)

    if lf_name[-1] == "s4":
        scale4 = np.zeros((sh[0], sh[1], np.int(0.25 * sh[2]), np.int(0.25 * sh[3]), sh[4]), np.float32)
        scale4[:, 4, :, :, :] = np.stack([cv.resize(LF_HR[i, 4, :, :, :], (np.int(0.25 * sh[3]), np.int(0.25 * sh[2])),
                                                    interpolation=cv.INTER_CUBIC) for i in range(0, sh[0])])
        scale4[4, :, :, :, :] = np.stack([cv.resize(LF_HR[4, i, :, :, :], (np.int(0.25 * sh[3]), np.int(0.25 * sh[2])),
                                                    interpolation=cv.INTER_CUBIC) for i in range(0, sh[0])])
        scale4_crosshair = np.zeros((2, sh[1], np.int(0.25 * sh[2]), np.int(0.25 * sh[3]), sh[4]), np.float32)
        scale4_crosshair[0, :, :, :, :] = scale4[:, 4, :, :, :]
        scale4_crosshair[1, :, :, :, :] = scale4[4, :, :, :, :]
        LF_gt['input'] = np.clip(scale4,0,1)
        LF_crosshair['input'] = np.clip(scale4_crosshair,0,1)
        cv_gt['input'] = np.clip(scale4[4,4,:,:,:],0,1)

    data = []

    color_space = hp.config['ColorSpace']

    if lf_name[-1] == 's2':
        LF_net_in = LF_gt['input_s2']
    else:
        LF_net_in = LF_gt['input_s4']

    if color_space == 'YCBCR':
        grey = False
        decoder_path = 'Y'

        result_lf = encode_decode_lightfield(data, LF_net_in,
                                             inputs, outputs, color_space, lf_name[-1],
                                             decoder_path=decoder_path, scales=scales)
        lf_out2 = dict()
        for i in range(0, len(scales)):
            lf_out2[scales[i]] = result_lf[2][scales[i]]
            mask = result_lf[1][scales[i]]
            lf_out2[scales[i]] = scale_back(lf_out2[scales[i]], mask)

        if grey:
            lf_YCBCR = dict()
            for i in range(0, len(scales)):
                sh1 = lf_out2[scales[i]].shape
                lf_YCBCR[scales[i]] = np.zeros((2, 9, sh1[2], sh1[3], 3), np.float32)
                lf_YCBCR[scales[i]][0, ...] = np.stack([cv.resize(rgb2YCbCr(LF_crosshair['input'][0, j, :, :, :]),
                                                                  (sh1[3], sh1[2]), interpolation=cv.INTER_CUBIC) for j
                                                        in range(0, 9)])
                lf_YCBCR[scales[i]][1, ...] = np.stack([cv.resize(rgb2YCbCr(LF_crosshair['input'][1, j, :, :, :]),
                                                                  (sh1[3], sh1[2]), interpolation=cv.INTER_CUBIC) for j
                                                        in range(0, 9)])

            lf_out3 = dict()
            for i in range(0, len(scales)):
                lf_out3[scales[i]] = np.concatenate((lf_out2[scales[i]], lf_YCBCR[scales[i]][:, :, :, :, 1:]), axis=-1)
        else:
            lf_out3 = lf_out2

        for i in range(0, len(scales)):
            tmp = np.zeros(lf_out3[scales[i]].shape, np.float32)
            tmp[0, ...] = np.stack(
                [np.clip(YCbCr2rgb(lf_out3[scales[i]][0,vv,:,:,:]), 0, 1) for vv in range(0, 9)])
            tmp[1, ...] = np.stack(
                [np.clip(YCbCr2rgb(lf_out3[scales[i]][1,vv,:,:,:]), 0, 1) for vv in range(0, 9)])
            lf_out3[scales[i]] = tmp


    else:
        decoder_path = 'RGB'

        result_lf = encode_decode_lightfield(data, LF_net_in,
                                             inputs, outputs, color_space, lf_name[-1],
                                             decoder_path=decoder_path, scales=scales)
        lf_out2 = dict()
        for i in range(0, len(scales)):
            lf_out2[scales[i]] = result_lf[2][scales[i]]
            mask = result_lf[1][scales[i]]
            lf_out2[scales[i]] = scale_back(lf_out2[scales[i]], mask)

        lf_out3 = lf_out2

    PSNR_out_lf = dict()
    SSIM_out_lf = dict()
    if 's2' in scales:
        # bring dimensions back
        LF_crosshair['input_s2'] = LF_crosshair['input_s2'][:,:,0:sh_orig_s2[2],0:sh_orig_s2[3],:]
        lf_out3['s2'] = lf_out3['s2'][:,:,0:sh_orig_s2[2],0:sh_orig_s2[3],:]

        sh = LF_crosshair['input_s2'].shape[2:]
        sx = hp.sx
        sy = hp.sy

        # LF_crosshair['input_s2'] = LF_crosshair['input_s2'][:, :, sx:sh[0] + 1 - sx, sy:sh[1] + 1 - sy, :]
        lf_out4 = dict()

        LF_crosshair['input_s2'] = LF_crosshair['input_s2'][:, :, sy:sh[0] + 1 - sy, sx:sh[1] + 1 - sx, :]
        lf_out4['s2'] = lf_out3['s2'][:, :, sy:sh[0] + 1 - sy, sx:sh[1] + 1 - sx, :]
        PSNR_out_lf['s2'] = np.mean(
            [measure.compare_psnr(LF_crosshair['input_s2'][i, j, :, :, :], lf_out4['s2'][i, j, :, :, :],
                                  data_range=1) for j in range(0, 9) for i in range(0, 2)])
        SSIM_out_lf['s2'] = np.mean(
            [measure.compare_ssim(LF_crosshair['input_s2'][i, j, :, :, :], lf_out4['s2'][i, j, :, :, :],
                                  data_range=1, multichannel=True) for j in range(0, 9) for i in range(0, 2)])

        print(np.mean(LF_crosshair['input_s2'] - lf_out4['s2']))

        plt.figure(1)
        plt.imshow(cv_gt['input_s2'])
        plt.show()

        plt.figure(2)
        for i in range(1, 10):
            plt.subplot(3, 3, i)
            if i == 1:
                plt.title("lf scale 2 vertical, psnr = %.2f\n ssim= %.2f" % (PSNR_out_lf['s2'], SSIM_out_lf['s2']))
            plt.imshow(lf_out4['s2'][0, i - 1, :, :, :])
        plt.show()

        plt.figure(3)
        for i in range(1, 10):
            plt.subplot(3, 3, i)
            if i == 1:
                plt.title(
                    "lf scale 2 horizontal, psnr = %.2f\n ssim= %.2f" % (PSNR_out_lf['s2'], SSIM_out_lf['s2']))
            plt.imshow(lf_out4['s2'][1, i - 1, :, :, :])
        plt.show()

    if 's4' in scales:
        if 'input_s4' in cv_gt:
            # bring dimensions back
            LF_crosshair['input_s4'] = LF_crosshair['input_s4'][:, :, 0:sh_orig_s4[2], 0:sh_orig_s4[3], :]
            lf_out3['s4'] = lf_out3['s4'][:, :, 0:sh_orig_s4[2], 0:sh_orig_s4[3], :]

            sh = LF_crosshair['input_s4'].shape[2:]

            sx = hp.sx
            sy = hp.sy

            LF_crosshair['input_s4'] = LF_crosshair['input_s4'][:, :, sy:sh[0] + 1 - sy, sx:sh[1] + 1 - sx, :]
            lf_out4['s4'] = lf_out3['s4'][:, :, sy:sh[0] + 1 - sy, sx:sh[1] + 1 - sx, :]
            PSNR_out_lf['s4'] = np.mean(
                [measure.compare_psnr(LF_crosshair['input_s4'][i, j, :, :, :], lf_out4['s4'][i, j, :, :, :],
                                      data_range=1) for j in range(0, 9) for i in range(0, 2)])
            SSIM_out_lf['s4'] = np.mean(
                [measure.compare_ssim(LF_crosshair['input_s4'][i, j, :, :, :], lf_out4['s4'][i, j, :, :, :],
                                      data_range=1, multichannel=True) for j in range(0, 9) for i in range(0, 2)])

            print(np.mean(LF_crosshair['input_s4'] - lf_out4['s4']))

        else:
            PSNR_out_lf['s4'] = 0
            SSIM_out_lf['s4'] = 0

            lf_out3['s4'] = lf_out3['s4'][:, :, 0:sh_orig_s4[2], 0:sh_orig_s4[3], :]

            lf_out4['s4'] = lf_out3['s4']

        plt.figure(2)
        for i in range(1, 10):
            plt.subplot(3, 3, i)
            if i == 1:
                plt.title(
                    "lf scale 4 vertical, psnr = %.2f\n ssim= %.2f" % (PSNR_out_lf['s4'], SSIM_out_lf['s4']))
            plt.imshow(lf_out4['s4'][0, i - 1, :, :, :])
        plt.show()

        plt.figure(2)
        for i in range(1, 10):
            plt.subplot(3, 3, i)
            if i == 1:
                plt.title(
                    "lf scale 4 horizontal, psnr = %.2f\n ssim= %.2f" % (PSNR_out_lf['s4'], SSIM_out_lf['s4']))
            plt.imshow(lf_out4['s4'][1, i - 1, :, :, :])
        plt.show()
        print('meow')



inputs.put( () )

