from matplotlib import pyplot as plt

import numpy as np
import scipy.io as sio

image = plt.imread('00350a_img.jpg')
bin_mask = sio.loadmat('00350a.mat')['person_segm'].astype('bool')

mask = np.dstack((bin_mask, bin_mask, bin_mask))

image[~mask] = 0
plt.imsave('final.png', image)
