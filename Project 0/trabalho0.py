from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pylab as plt


# 1.1
def ajuste_brilho(A, gama):
    return ((A * (1 / 255)) ** (1 / gama)) / (1 / 255)


# 1.2
def plano_bits(A, ordem):
    return np.floor(A / 2 ** ordem) % 2


# 1.4
def combinacao_img(A, per_A, B, per_B):
    return A * per_A + B * per_B


A = misc.imread('baboon.png')
B = misc.imread('butterfly.png')

plt.imshow(plano_bits(A, 0), cmap='gray')
plt.show()
