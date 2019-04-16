import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from libs.convert_colorspace import rgb2YUV,rgb2YCbCr
from scipy.misc import imread
from skimage.color import rgb2lab, lab2rgb
a = np.zeros((2,512,512,3))
imgx = imread('/home/z/PycharmProjects/SR/full_data_512/platonic/input_Cam044.png')/255
imgy = imread('/home/z/PycharmProjects/SR/full_data_512/platonic/input_Cam045.png')/255
a[0,:,:,:] = imgx
a[1,:,:,:] = imgy
a2 = rgb2lab(a)
img1x = rgb2lab(imgx)
img1y = rgb2lab(imgy)
err1 = np.sum(np.abs(img1x - a2[0,:,:,:]))
err2 = np.sum(np.abs(img1y - a2[1,:,:,:]))
img2 = rgb2YUV(imgx)
err = np.sum(np.abs(img1x - img2))
k = 0