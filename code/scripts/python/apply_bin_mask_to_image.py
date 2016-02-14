from matplotlib import pyplot as plt

import numpy as np
import scipy.io as sio

image = plt.imread('input.png')
bin_mask = sio.loadmat('mask.mat')['bin_mask'].astype('bool')

mask = np.dstack((bin_mask, bin_mask, bin_mask))

image[~mask] = 0
plt.imsave('output.png', image)
