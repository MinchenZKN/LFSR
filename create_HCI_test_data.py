import numpy as np
import scipy.io as sc
import h5py
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from file_io import get_dataset_as_type



data_source = "/home/mz/HD_data/SR_data_backups/HCI LF/gantry/"
# data = sc.loadmat(data_path)
name_file = '/lf.h5'
data_folders = []
# data_folders.append('buddha')
# data_folders.append('buddha2')
# data_folders.append('horses')
# data_folders.append('medieval')
# data_folders.append('monasRoom')
# data_folders.append('papillon')
# data_folders.append('stillLife')
data_folders.append('couple')
data_folders.append('cube')
data_folders.append('pyramide')
data_folders.append('statue')
data_folders.append('transparency')

test_data_dir = "/home/mz/HD_data/SR_data_backups/HCI test/"
test_data_filename = 'lf_test_HCI_'

scale = 4
nviews = 9
size = 1024

for i in range(0,len(data_folders)):
    data_path = os.path.join(data_source, data_folders[i]) + name_file
    f = h5py.File(data_path, 'r')
    # testf = f['LF']
    # LF = get_dataset_as_type(f['LF'], dtype='float32')

    LF_temp = f['LF'].value
    LF_temp = LF_temp.astype(np.float32) / 255.0
    LF_temp = np.flip(LF_temp, axis=1)


    file = h5py.File(test_data_dir + test_data_filename + data_folders[i] + '.hdf5', 'w')
    print('generating LF file %s' %data_folders[i])


    dset_LF = file.create_dataset('LF', data=LF_temp)

    # next dataset
    print(' done.')
















