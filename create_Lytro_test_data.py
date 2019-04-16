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

# patch size. patches of this size will be extracted and stored
# must remain fixed, hard-coded in NN

s4 = 4
s2 = 2

px_LR_s4 = 48
py_LR_s4 = 48
px_LR_s2 = int(px_LR_s4 * s2)
py_LR_s2 = int(py_LR_s4 * s2)
px = int(px_LR_s4 * s2)
py = int(py_LR_s4 * s2)
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
sx = int(sx_LR_s4 * s2)
sy = int(sy_LR_s4 * s2)



# output file to write to
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! careful: overwrite mode !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# previous training data will be erased.

training_data_dir = "/home/mz/HD_data/SR_data_backups/LYTRO test/"
training_data_filename = 'lf_test_lytro_'


data_source = "/home/mz/HD_data/SR_data_backups/lytro/"
# data = sc.loadmat(data_path)
name_ending = '_lf.mat'


# data_source = "/home/mz/HD data/test brightness/"
data_folders = os.listdir(data_source)

index = 0
idx_folder = 0
for lf_name in data_folders:

    file = h5py.File(training_data_dir + training_data_filename+lf_name[0:-7]+'.hdf5', 'w')
    data_folder = os.path.join(data_source, lf_name)
    print("now %i / %i" % (idx_folder+1, len(data_folders)))
    idx_folder = idx_folder+1

    data_path = data_folder
    f = h5py.File(data_path, 'r')

    LF_temp = np.transpose(f['LF'], (4, 3, 2, 1, 0))
    LF_temp = LF_temp.astype(np.float32)

    dset_LF = file.create_dataset('LF', data=LF_temp)
    # next dataset
    print(' done.')

