import numpy as np
import scipy.io as sc
import h5py
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from file_io import get_dataset_as_type
import lf_tools
import file_io


# data_source = "/home/mz/HD data/SR data backups/full_data_512/"
# data_folders = os.listdir(data_source)
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
# for lf_name in data_folders:
#
#     data_folder = os.path.join(data_source, lf_name)
#
#     LF = file_io.read_lightfield(data_folder)
#     LF = LF.astype(np.float32)
#
#     (stack_h, stack_v) = lf_tools.epi_stacks(LF, 0, 0, 512, 512)
    # stack_h = np.transpose(stack_h, (0, 2, 1, 3))
#

data_source = "/home/mz/PyCharm/Data/testData_s4/super_resolution/HCI/not seen/lf_test_HCI_"
# data = sc.loadmat(data_path)
name_file = '.hdf5'
data_folders = []
data_folders.append('buddha')
# data_folders.append('buddha2')
# data_folders.append('horses')
# data_folders.append('medieval')
# data_folders.append('monasRoom')
# data_folders.append('papillon')
# data_folders.append('stillLife')
for i in range(0,len(data_folders)):

    data_path = data_source + data_folders[i] + name_file
    f = h5py.File(data_path, 'r')

    LF = f['LF'].value
    LF_LR = f['LF_LR'].value


    (stack_h, stack_v) = lf_tools.epi_stacks(LF, 0, 0, LF.shape[2], LF.shape[3])
    (stack_h2, stack_v2) = lf_tools.epi_stacks(LF_LR, 0, 0, LF_LR.shape[2], LF_LR.shape[3])
    # stack_h = np.transpose(stack_h, (0, 2, 1, 3))
    # for k in range(0, 9):
    #     plt.figure(k)
    #     plt.imshow(stack_h[k, :, :, :])
k = 0