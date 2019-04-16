#
# Push a light field through decoder/encoder modules of the autoencoder
#

from queue import Queue
import code
import numpy as np
from libs.convert_colorspace import rgb2YCbCr, YCbCr2rgb, rgb2YUV
from scipy.misc import imresize
import cv2 as cv
import st
from scipy.interpolate import interpn

import scipy
import scipy.signal
from scipy.misc import imresize

# timing and multithreading
import _thread
import time
from timeit import default_timer as timer
import matplotlib.pyplot as plt
# light field GPU tools
import lf_tools

from skimage.color import rgb2lab
from libs.convert_colorspace import rgb2YCbCr, rgb2YUV
import cv2


# data config
import config_data_format as cdf
import interpolate_mask as im
import config_autoencoder_rgb_3d as hp

def add_result_to_cv( data, result, LF_crosshair,  mask_sum, bs_x, bs_y, bxx, scale, colorspace):

  """ note: numpy arrays are passed by reference ... I think
  """
  H_mask = hp.eval_res['h_mask_'+scale]
  W_mask = hp.eval_res['w_mask_'+scale]

  m = hp.eval_res['m_'+scale]

  print( 'x', end='', flush=True )
  by = result[1]['py']
  sv_v = result[0]['SR_v_'+scale]
  sv_h = result[0]['SR_h_' + scale]
  H_patch = sv_v.shape[-2]

  if colorspace == 'YCBCR':
    sv_v = np.clip(sv_v, 16.0 / 255.0, 240.0 / 255.0)
    sv_v[...,0] = np.clip(sv_v[...,0], 16.0 / 255.0, 235.0 / 255.0)
    sv_h = np.clip(sv_h, 16.0 / 255.0, 240.0 / 255.0)
    sv_h[...,0] = np.clip(sv_h[...,0], 16.0 / 255.0, 235.0 / 255.0)
  else:
    sv_v = np.clip(sv_v,0,1)
    sv_h = np.clip(sv_h, 0, 1)

  num_channels = LF_crosshair.shape[-1]

  mask = im.get_mask(H_mask,W_mask,m)
  mask3d = np.expand_dims(mask, axis = 2)
  mask3d = np.tile(mask3d, (1, 1, num_channels))
  maskLF = np.expand_dims(mask3d, axis=3)
  maskLF = np.transpose(np.tile(maskLF, (1, 1, 1, 9)), [3, 0, 1, 2])

  # cv data is in the center of the result stack
  # lazy, hardcoded the current fixed size
  if scale == 's2':
    p = H_mask//2 - hp.sy_s2//2
    q = H_mask//2 + hp.sy_s2//2
    sx = hp.sx_s2
    sy = hp.sy_s2
  elif scale == 's4':
    p = H_mask//2 - hp.sy_HR//2
    q = H_mask//2 + hp.sy_HR//2
    sx = hp.sx_HR
    sy = hp.sy_HR


  for bx in range(bxx):
    px = bs_x * bx + sx
    py = bs_y * by + sy

    LF_crosshair[0, :, py - p:py + q, px - p:px + q, :] = np.add(LF_crosshair[0, :, py - p:py + q, px - p:px + q, :]
                                                                 , np.multiply(
        sv_v[bx, :, H_patch // 2 - H_mask // 2: H_patch // 2 + H_mask // 2,
        H_patch // 2 - H_mask // 2: H_patch // 2 + H_mask // 2, :], maskLF))

    LF_crosshair[1, :, py - p:py + q, px - p:px + q, :] = np.add(LF_crosshair[1, :, py - p:py + q, px - p:px + q, :]
                                                                 , np.multiply(
        sv_h[bx, :, H_patch // 2 - H_mask // 2: H_patch // 2 + H_mask // 2,
        H_patch // 2 - H_mask // 2: H_patch // 2 + H_mask // 2, :], maskLF))

    mask_sum[py-p:py+q , px-p:px+q] = mask_sum[py-p:py+q , px-p:px+q] + mask


def encode_decode_lightfield(data, LF, inputs, outputs, ColorSpace, lf_scale, decoder_path, scales):
  # light field size
  H = LF.shape[2]
  W= LF.shape[3]

  if lf_scale =='s2':
    # patch step sizes
    bs_y = hp.sy_s2
    bs_x = hp.sx_s2
    # patch height/width
    ps_y = hp.H_s2
    ps_x = hp.W_s2
    ps_v = hp.D

  if lf_scale =='s4':
    # patch step sizes
    bs_y = hp.sy_HR
    bs_x = hp.sx_HR
    # patch height/width
    ps_y = hp.H_s4
    ps_x = hp.W_s4
    ps_v = hp.D

  # patches per row/column
  by = np.int16((H - ps_y) / bs_y) + 1
  bx = np.int16((W - ps_x) / bs_x) + 1


  num_channels = hp.decoders_3D[0]['channels']

  print('starting LF encoding/decoding [', end='', flush=True)
  start = timer()

  # one complete row per batch
  mask_sum = dict()
  LF_crosshair = dict()
  if lf_scale == 's2':
    if 's2' in scales:
      mask_sum['s2'] = np.zeros([np.int(H), np.int(W)], dtype=np.float32)
      LF_crosshair['s2'] = np.zeros([2, 9, np.int(H), np.int(W), num_channels], dtype=np.float32)
    if 's4' in scales:
      mask_sum['s4'] = np.zeros([np.int(2*H), np.int(2*W)], dtype=np.float32)
      LF_crosshair['s4'] = np.zeros([2, 9, np.int(2*H), np.int(2*W), num_channels], dtype=np.float32)

  if lf_scale == 's4':
    if 's2' in scales:
      mask_sum['s2'] = np.zeros([np.int(H/2), np.int(W/2)], dtype=np.float32)
      LF_crosshair['s2'] = np.zeros([2, 9, np.int(H/2), np.int(W/2), num_channels], dtype=np.float32)
    if 's4' in scales:
      mask_sum['s4'] = np.zeros([np.int(H), np.int(W)], dtype=np.float32)
      LF_crosshair['s4'] = np.zeros([2, 9, np.int(H), np.int(W), num_channels], dtype=np.float32)

  results_received = 0
  for py in range(by):
    print('.', end='', flush=True)
    batch = dict()
    batch['stacks_h_HR'] = np.zeros([bx, ps_v, ps_y, ps_x, hp.C], np.float32)
    batch['stacks_v_HR'] = np.zeros([bx, ps_v, ps_y, ps_x, hp.C], np.float32)
    if batch['stacks_v_HR'].shape[-2] == 192:
      batch['stacks_v_s4'] = np.zeros([bx] + [9, 192, 192, 3], np.float32)
      batch['stacks_h_s4'] = np.zeros([bx] + [9, 192, 192, 3], np.float32)
      batch['stacks_v_s2'] = np.zeros([bx] + [9, 96, 96, 3], np.float32)
      batch['stacks_h_s2'] = np.zeros([bx] + [9, 96, 96, 3], np.float32)
      batch['stacks_v'] = np.zeros([bx] + [9, 48, 48, 3], np.float32)
      batch['stacks_h'] = np.zeros([bx] + [9, 48, 48, 3], np.float32)
      batch['stacks_bicubic_v'] = np.zeros([bx] + [9, 48, 48, 3], np.float32)
      batch['stacks_bicubic_h'] = np.zeros([bx] + [9, 48, 48, 3], np.float32)
    if batch['stacks_v_HR'].shape[-2] == 96:
      batch['stacks_v_s2'] = np.zeros([bx] + [9, 96, 96, 3], np.float32)
      batch['stacks_h_s2'] = np.zeros([bx] + [9, 96, 96, 3], np.float32)
      batch['stacks_v'] = np.zeros([bx] + [9, 48, 48, 3], np.float32)
      batch['stacks_h'] = np.zeros([bx] + [9, 48, 48, 3], np.float32)
      batch['stacks_bicubic_v'] = np.zeros([bx] + [9, 48, 48, 3], np.float32)
      batch['stacks_bicubic_h'] = np.zeros([bx] + [9, 48, 48, 3], np.float32)
    for px in range(bx):
      # get single patch
      patch = cdf.get_patch(LF, py, px, lf_scale)
      if ColorSpace == 'YCBCR':
        patch['stack_v_HR'] = rgb2YCbCr(np.clip(patch[ 'stack_v_HR' ] , 0.0, 1.0))
        patch['stack_h_HR'] = rgb2YCbCr(np.clip(patch[ 'stack_h_HR' ] , 0.0, 1.0))
        if 'stacks_v_s4' in batch:
          batch['stacks_v_s4'][px, ...] = np.clip(patch['stack_v_HR'], 16.0 / 255.0, 240 / 255)
          batch['stacks_h_s4'][px, ...] = np.clip(patch['stack_h_HR'], 16.0 / 255.0, 240 / 255)

          batch['stacks_v_s2'][px, ...] = np.clip(np.stack([cv2.resize(patch['stack_v_HR'][i, :, :, :], (96, 96),
                                                                   interpolation=cv2.INTER_CUBIC) for i in
                                                        range(0, 9)]), 16.0 / 255.0, 240 / 255)
          batch['stacks_h_s2'][px, ...] = np.clip(np.stack([cv2.resize(patch['stack_h_HR'][i, :, :, :], (96, 96),
                                                                   interpolation=cv2.INTER_CUBIC) for i in
                                                        range(0, 9)]), 16.0 / 255.0, 240 / 255)
          batch['stacks_v'][px, ...] = np.clip(np.stack([cv2.resize(patch['stack_v_HR'][i, :, :, :], (48, 48),
                                                                interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]),
                                           16.0 / 255.0, 240 / 255)
          batch['stacks_h'][px, ...] = np.clip(np.stack([cv2.resize(patch['stack_h_HR'][i, :, :, :], (48, 48),
                                                                interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]),
                                           16.0 / 255.0, 240 / 255)

        else:
          batch['stacks_v_s2'][px, ...] = np.clip(patch['stack_v_HR'], 16.0 / 255.0, 240 / 255)
          batch['stacks_h_s2'][px, ...] = np.clip(patch['stack_h_HR'], 16.0 / 255.0, 240 / 255)
          batch['stacks_v'][px, ...] = np.clip(
            np.stack([cv2.resize(patch['stack_v_HR'][i, :, :, :], (48, 48),
                                 interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]), 16.0 / 255.0, 240 / 255)
          batch['stacks_h'][px, ...] = np.clip(
            np.stack([cv2.resize(patch['stack_h_HR'][i, :, :, :], (48, 48),
                                 interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]), 16.0 / 255.0, 240 / 255)

        # biclinear stuff
        [tq, vq, uq] = np.meshgrid(range(9), range(48), range(48))
        tq = tq.transpose(1, 0, 2)
        vq = vq.transpose(1, 0, 2)
        uq = uq.transpose(1, 0, 2)
        points = [[0, 4, 8], np.arange(48), np.arange(48)]
        xi = (tq.ravel(), vq.ravel(), uq.ravel())
        for ch in range(0, 3):
          batch['stacks_bicubic_v'][px, :, :, :, ch] = np.clip(
            interpn(points, batch['stacks_v'][px, 0:9:4, :, :, ch], xi).reshape((9, 48, 48)), 16.0 / 255.0,
                                                                                              240 / 255)
          batch['stacks_bicubic_h'][px, :, :, :, ch] = np.clip(
            interpn(points, batch['stacks_h'][px, 0:9:4, :, :, ch],
                    xi).reshape((9, 48, 48)), 16.0 / 255.0, 240 / 255)

      if ColorSpace == 'RGB':
        patch['stack_v_HR'] = np.clip(patch[ 'stack_v_HR' ] , 0.0, 1.0)
        patch['stack_h_HR'] = np.clip(patch[ 'stack_h_HR' ] , 0.0, 1.0)
        if 'stacks_v_s4' in batch:
          batch['stacks_v_s4'][px, ...] = np.clip(patch['stack_v_HR'], 0.0, 1.0)
          batch['stacks_h_s4'][px, ...] = np.clip(patch['stack_h_HR'], 0.0, 1.0)

          batch['stacks_v_s2'][px, ...] = np.clip(np.stack([cv2.resize(patch['stack_v_HR'][i, :, :, :], (96, 96),
                                                                   interpolation=cv2.INTER_CUBIC) for i in
                                                        range(0, 9)]), 0.0, 1.0)
          batch['stacks_h_s2'][px, ...] = np.clip(np.stack([cv2.resize(patch['stack_h_HR'][i, :, :, :], (96, 96),
                                                                   interpolation=cv2.INTER_CUBIC) for i in
                                                        range(0, 9)]), 0.0, 1.0)
          batch['stacks_v'][px, ...] = np.clip(np.stack([cv2.resize(patch['stack_v_HR'][i, :, :, :], (48, 48),
                                                                interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]),
                                               0.0, 1.0)
          batch['stacks_h'][px, ...] = np.clip(np.stack([cv2.resize(patch['stack_h_HR'][i, :, :, :], (48, 48),
                                                                interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]),
                                               0.0, 1.0)

        else:
          batch['stacks_v_s2'][px, ...] = np.clip(patch['stack_v_HR'], 0.0, 1.0)
          batch['stacks_h_s2'][px, ...] = np.clip(patch['stack_h_HR'], 0.0, 1.0)
          batch['stacks_v'][px, ...] = np.clip(
            np.stack([cv2.resize(patch['stack_v_HR'][i, :, :, :], (48, 48),
                                 interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]), 0.0, 1.0)
          batch['stacks_h'][px, ...] = np.clip(
            np.stack([cv2.resize(patch['stack_h_HR'][i, :, :, :], (48, 48),
                                 interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]), 0.0, 1.0)

        # biclinear stuff
        [tq, vq, uq] = np.meshgrid(range(9), range(48), range(48))
        tq = tq.transpose(1, 0, 2)
        vq = vq.transpose(1, 0, 2)
        uq = uq.transpose(1, 0, 2)
        points = [[0, 4, 8], np.arange(48), np.arange(48)]
        xi = (tq.ravel(), vq.ravel(), uq.ravel())
        for ch in range(0, 3):
          batch['stacks_bicubic_v'][px, :, :, :, ch] = np.clip(
            interpn(points, batch['stacks_v'][px, 0:9:4, :, :, ch], xi).reshape((9, 48, 48)), 0.0, 1.0)
          batch['stacks_bicubic_h'][px, :, :, :, ch] = np.clip(
            interpn(points, batch['stacks_h'][px, 0:9:4, :, :, ch],
                    xi).reshape((9, 48, 48)), 0.0, 1.0)

    # push complete batch to encoder/decoder pipeline
    batch['py'] = py
    batch['decoder_path'] = decoder_path

    inputs.put(batch)

    #
    if not outputs.empty():
      result = outputs.get()
      if 's2' in scales:
        add_result_to_cv(data, result, LF_crosshair['s2'], mask_sum['s2'], hp.sx_s2, hp.sy_s2, bx, 's2', ColorSpace)
      if 's4' in scales:
        add_result_to_cv(data, result, LF_crosshair['s4'], mask_sum['s4'], hp.sx_HR, hp.sy_HR, bx,'s4', ColorSpace)
      results_received += 1
      outputs.task_done()

  # catch remaining results

  while results_received < by:
    result = outputs.get()
    if 's2' in scales:
      add_result_to_cv(data, result, LF_crosshair['s2'], mask_sum['s2'], hp.sx_s2, hp.sy_s2, bx, 's2', ColorSpace)
    if 's4' in scales:
      add_result_to_cv(data, result, LF_crosshair['s4'], mask_sum['s4'], hp.sx_HR, hp.sy_HR, bx, 's4', ColorSpace)
    results_received += 1
    outputs.task_done()

  # elapsed time since start of dmap computation
  end = timer()
  total_time = end - start
  print('] done, total time %g seconds.' % total_time)

  # evaluate result
  # mse = 0.0

  # compute stats and return result
  print('total time ', end - start)
  # print('MSE          : ', mse)

  # code.interact( local=locals() )
  return (total_time, mask_sum, LF_crosshair)

def scale_back(im, mask):
  H = mask.shape[0]
  W = mask.shape[1]
  mask[mask == 0] = 1
  num_channels = im.shape[-1]

  if len(im.shape) == 3:
    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1, 1, num_channels))

  if len(im.shape) == 5:
    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1, 1, num_channels))

    mask = np.expand_dims(mask, axis = 3)
    mask = np.transpose(np.tile(mask, (1, 1, 1, 9)), [3, 0, 1, 2])
    mask1 = np.zeros((2,9,H,W,num_channels), dtype = np.float32)
    mask1[0,:,:,:,:] = mask
    mask1[1, :, :, :, :] = mask
    del mask
    mask = mask1

  return(np.divide(im,mask))