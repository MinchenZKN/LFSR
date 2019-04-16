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
# OUTPUT CONFIGURATION
from file_io import read_img
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

test_data_dir = "/home/mz/HD_data/SR_data_backups/STANFORD test/"
test_data_filename = 'lf_test_stanford_HR_'
#
# data_folders = ( ( "training", "boxes" ), )
# data_folders = data_folders_base + data_folders_add
data_source = "/home/mz/HD_data/SR_data_backups/stanford HCI train/STF/"
# data_folders = os.listdir(data_source)
data_folders = []
data_folders.append('chess/')
# data_folders.append('Truck')
# data_folders.append('Treasure')
# data_folders.append('Bulldozer')
# data_folders.append('JellyBeans')
# data_folders.append('Eucalyptus')


#
# loop over all datasets, write out each dataset in patches
# to feed to autoencoder in random order
#

for lf_name in data_folders:
    file = h5py.File(test_data_dir + test_data_filename + lf_name + '.hdf5', 'w')
    data_folder = os.path.join(data_source, lf_name)
    # read diffuse color
    print("now %s" % (lf_name))

    data_path = data_folder
    ims = sorted(os.listdir(data_path))
    img = read_img(data_path + '/' + ims[0])
    sh = img.shape
    LF_temp = np.zeros([9, 9, sh[0], sh[1], sh[2]])

    for v in range(0, nviews):
        for h in range(0, nviews):
            ind_img = int(v * nviews + h)
            image = read_img(data_path + '/' + ims[ind_img])
            LF_temp[v, h, :, :, :] = np.divide(image, 255)

    LF_temp = LF_temp.astype(np.float32)
    LF_temp = np.flip(LF_temp, axis=0)
    dset_LF = file.create_dataset('LF', data=LF_temp)

    # next dataset
    print(' done.')
