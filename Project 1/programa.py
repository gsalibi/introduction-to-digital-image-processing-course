import cv2
from matplotlib import pyplot as plt
import numpy as np

image1 = plt.imread('baboon.png', 0)
image2 = plt.imread('house.png', 0)
image = image2  # seleciona qual imagem ser￿á trabalhada


# 1.1
# cria filtros e aplica ￿à imagem

h1 = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])  # cria h1
h1 = cv2.filter2D(image,-1,h1)  # aplica h1 à imagem

h2 = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])/256 # cria h2
h2 = cv2.filter2D(image,-1,h2)  # aplica h2 à imagem

h3 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) # cria h3
h3 = cv2.filter2D(image,-1,h3)  # aplica h3 à imagem

h4 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) # cria h4
h4 = cv2.filter2D(image,-1,h4)  # aplica h4 à imagem

h5 = np.hypot(h3, h4) # cria h5
h5 = np.ndarray.astype(h5, float)  # aplica h5 à imagem

# Exibe imagem final na tela e salva
plt.imshow(image, cmap='gray')  # original
plt.show()
plt.imshow(h1, cmap='gray')  # h1
plt.savefig('imagem1-1-h1.png')
plt.show()
plt.imshow(h2, cmap='gray')  # h2
plt.savefig('imagem1-1-h2.png')
plt.show()
plt.imshow(h3, cmap='gray')  # h3
plt.savefig('imagem1-1-h3.png')
plt.show()
plt.imshow(h4, cmap='gray')  # h4
plt.savefig('imagem1-1-h4.png')
plt.show()
plt.imshow(h5, cmap='gray')  # h5
plt.savefig('imagem1-1-h5.png')
plt.show()


# 1.2

# Expectro de Fourrier
fshift = np.fft.fft2(image)  # converte imagem para Fourier
fshift = np.fft.fftshift(fshift)  # deloca a frequência-zero para o centro


# Cria filtro gaussiano
gf = cv2.getGaussianKernel(image.shape[0], sigma=40)
gf = gf * gf.T
# Aplica filtro
filtImg40 = fshift * gf

# repete para outros valores de sigma
gf = cv2.getGaussianKernel(image.shape[0], sigma=10)
gf = gf * gf.T
# Aplica filtro
filtImg10 = fshift * gf

# repete para outros valores de sigma
gf = cv2.getGaussianKernel(image.shape[0], sigma=5)
gf = gf * gf.T
# Aplica filtro
filtImg5 = fshift * gf

# Recupera imagem
res40 = np.fft.ifftshift(filtImg40)  # volta o deslocamento da frequência-zero
res40 = np.fft.ifft2(res40)  # reverte Fourier
res40 = np.abs(res40)  # converte o tipo para poder utilizar
res10 = np.fft.ifftshift(filtImg10)  # volta o deslocamento da frequência-zero
res10 = np.fft.ifft2(res10)  # reverte Fourier
res10 = np.abs(res10)  # converte o tipo para poder utilizar
res5 = np.fft.ifftshift(filtImg5)  # volta o deslocamento da frequência-zero
res5 = np.fft.ifft2(res5)  # reverte Fourier
res5 = np.abs(res5)  # converte o tipo para poder utilizar

# Exibe imagem final na tela e salva
plt.imshow(res40, cmap='gray')
plt.savefig('imagem1-2-sigma-40.png')
plt.show()
plt.imshow(res10, cmap='gray')
plt.savefig('imagem1-2-sigma-10.png')
plt.show()
plt.imshow(res5, cmap='gray')
plt.savefig('imagem1-2-sigma-5.png')
plt.show()
