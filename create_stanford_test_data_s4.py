import numpy as np
import scipy.io as sc
import h5py
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter




data_source = "/home/mz/HD data/SR data backups/STANFORDDATA/"
# data = sc.loadmat(data_path)
name_ending = '_lf.mat'
data_names = []
# data_names.append('Bunny')
# data_names.append('Eucalyptus')
# data_names.append('JellyBeans')
# data_names.append('LegoBulldozer')
# data_names.append('LegoTruck')
data_names.append('TreasureChest')

test_data_dir = "/home/mz/PyCharm/Data/testData_s4/super_resolution/stanford/not seen/"
test_data_filename = 'lf_test_stanford_'

scale = 4
nviews = 9
size = 1024

for i in range(0,len(data_names)):
    data_path = os.path.join(data_source, data_names[i]) + name_ending
    f = h5py.File(data_path, 'r')
    # testf = f['LF']
    LF = np.transpose(f['LF'], (4, 3, 2, 1, 0))
    file = h5py.File(test_data_dir + test_data_filename + data_names[i] + '.hdf5', 'w')
    print('generating LF file %s' %data_names[i])
    LF = LF.astype(np.float32)

    LF_temp = np.transpose(f['LF'], (4, 3, 2, 1, 0))
    LF_temp = LF_temp.astype(np.float32)

    LF_LR = np.zeros((LF.shape[0], LF.shape[1], int(LF.shape[2] / scale),
                      int(round(LF.shape[3] / scale)), int(LF.shape[4])), np.float32)

    for v in range(0, nviews):
        for h in range(0, nviews):
            LF[v, h, :, :, :] = gaussian_filter(LF[v, h, :, :, :], sigma=0.7, truncate=2)
            LF_LR[v, h, :, :, :] = LF[v, h, 0:LF.shape[2] - 1:scale, 0:LF.shape[3] - 1:scale, :]

    dset_LF_LR = file.create_dataset('LF_LR', data=LF_LR)
    dset_LF = file.create_dataset('LF', data=LF_temp)

    # next dataset
    print(' done.')










# f =  h5py.File(data_path, 'r')
# a = f['LF']
# a = np.transpose(a, (4, 3, 2, 1, 0))
# plt.imshow(a[4,4,:,:,:])
# plt.show()





