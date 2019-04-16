import numpy as np
import scipy.io as sc
import h5py
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from file_io import get_dataset_as_type



data_source = "/home/mz/HD data/SR data backups/HCI LF/blender/"
# data = sc.loadmat(data_path)
name_file = '/lf.h5'
data_folders = []
# data_folders.append('buddha')
# data_folders.append('buddha2')
# data_folders.append('horses')
# data_folders.append('medieval')
# data_folders.append('monasRoom')
# data_folders.append('papillon')
data_folders.append('stillLife')

test_data_dir = "/home/mz/PyCharm/Data/testData_s4/super_resolution/HCI/not seen/"
test_data_filename = 'lf_test_HCI_'

scale = 4
nviews = 9
size = 1024

for i in range(0,len(data_folders)):
    data_path = os.path.join(data_source, data_folders[i]) + name_file
    f = h5py.File(data_path, 'r')
    # testf = f['LF']
    # LF = get_dataset_as_type(f['LF'], dtype='float32')
    LF = f['LF'].value
    LF = LF.astype(np.float32)/255.0
    LF = np.flip(LF, axis=1)
    LF_temp = f['LF'].value
    LF_temp = LF_temp.astype(np.float32) / 255.0
    LF_temp = np.flip(LF_temp, axis=1)
    file = h5py.File(test_data_dir + test_data_filename + data_folders[i] + '.hdf5', 'w')
    print('generating LF file %s' %data_folders[i])


    LF_LR = np.zeros((LF.shape[0], LF.shape[1], int(LF.shape[2] / scale),
                      int(LF.shape[3] / scale), int(LF.shape[4])), np.float32)

    for v in range(0, nviews):
        for h in range(0, nviews):
            LF[v, h, :, :, :] = gaussian_filter(LF[v, h, :, :, :], sigma=0.7, truncate=2)
            LF_LR[v, h, :, :, :] = LF[v, h, 0:LF.shape[2] - 1:scale, 0:LF.shape[3] - 1:scale, :]

    dset_LF_LR = file.create_dataset('LF_LR', data=LF_LR)
    dset_LF = file.create_dataset('LF', data=LF_temp)

    # next dataset
    print(' done.')
















