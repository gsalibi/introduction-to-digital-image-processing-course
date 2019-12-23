import cv2
import numpy as np
from math import floor

# cria imagem com pontilhado ordenado
def pontilhado_ordenado(img, matriz, dimensoes):
    i, j = img.shape
    saida = np.array(i * j * matriz)
    saida = saida.reshape(i, j, dimensoes, dimensoes).swapaxes(1, 2).reshape(i*dimensoes, j*dimensoes)
    x, y = saida.shape
    for i in range(x):
        for j in range(y):
            if saida[i, j] < img[floor(i / dimensoes), floor(j / dimensoes)]:
                saida[i, j] = 255
            else:
                saida[i, j] = 0

    return saida.astype(np.uint8)


# cria imagem com a técnica de Floyd-Steinberg
def pontilhado_floyd_steinberg(img, zigzag=True):
    img = img.astype(np.int16)
    saida = np.zeros_like(img).astype(np.uint8)
    h, w = img.shape
    limiar = lambda x: 255 * floor(x / 128)
    for i in range(h):
        if zigzag:
            rev = i % 2 == 1
        else:
            rev = False
        for j in range(w)[::-1 if rev else 1]:
            # configura pixel na imagem de saída
            saida[i,j] = limiar(img[i,j])
            # propaga erro
            erro = img[i,j] - saida[i,j]
            if not rev:
                if j+1 < w:
                    img[i,j+1] += round(7/16 * erro)
                if j+1 < w and i+1 < h:
                    img[i+1,j+1] += round(1/16 * erro)
                if i+1 < h:
                    img[i+1,j] += round(5/16 * erro)
                if i+1 < h and j-1 >= 0:
                    img[i+1,j-1] += round(3/16 * erro)
            else:
                if j-1 >= 0:
                    img[i,j-1] += round(7/16 * erro)
                if j+1 < w and i+1 < h:
                    img[i+1,j+1] += round(3/16 * erro)
                if i+1 < h:
                    img[i+1,j] += round(5/16 * erro)
                if i+1 < h and j-1 >= 0:
                    img[i+1,j-1] += round(1/16 * erro)
    return saida


# cria matrizes especificadas no trabalho
matriz_3x3 = [[6, 8, 4],
              [1, 0, 3],
              [5, 2, 7]]
matriz_bayer = [[0, 12, 3, 15],
                [8, 4, 11, 7],
                [2, 14, 1, 13],
                [10, 6, 9, 5]]
    
# imprime menu
print('''
Selecione a imagem a ser trabalhada:
    1 - baboon.pgm
    2 - fiducial.pgm
    3 - monarch.pgm
    4 - peppers.pgm
    5 - retina.pgm
    6 - sonnet.pgm
    7 - wedge.pgm
    8 - lena.pgm
        ''')

# seleciona opção
op = 0
while op == 0:
    op = int(input())
    if op == 1:
        img_nome = "baboon"
    elif op == 2:
        img_nome = "fiducial"
    elif op == 3:
        img_nome = "monarch"
    elif op == 4:
        img_nome = "peppers"
    elif op == 5:
        img_nome = "retina"
    elif op == 6:
        img_nome = "sonnet"
    elif op == 7:
        img_nome = "wedge"
    elif op == 8:
        img_nome = "lena"
    else:
        print("Opção inválida. Tente novamente.")
        op = 0
print("processando imagens...\n")

# abre imagem
img_caminho = "imagens/entrada/" + img_nome + ".pgm"
img = cv2.imread(img_caminho, cv2.IMREAD_GRAYSCALE)

# matriz 3x3
img_3x3 = np.round(img / np.max(img) * 9).astype(np.uint8) # normaliza para o tamanho da matriz
saida = pontilhado_ordenado(img_3x3, matriz_3x3, 3)
cv2.imwrite("imagens/saída/" + img_nome + "-3x3.pgm", saida)

# matriz de Bayer
img_bayer = np.round(img / np.max(img) * 16).astype(np.uint8) # normaliza para o tamanho da matriz
saida = pontilhado_ordenado(img_bayer, matriz_bayer, 4)
cv2.imwrite("imagens/saída/" + img_nome + "-bayer.pgm", saida)

# Floyd-Steinberg com zig zag
saida = pontilhado_floyd_steinberg(img, zigzag=True)
cv2.imwrite("imagens/saída/" + img_nome + "-Floyd-Steinberg-com-zig-zag.pgm", saida)

# Floyd-Steinberg sem zig zag
saida = pontilhado_floyd_steinberg(img, zigzag=False)
cv2.imwrite("imagens/saída/" + img_nome + "-Floyd-Steinberg-sem-zig-zag.pgm", saida)

# finaliza programa
print("Tudo certo! Imagens salvas na pasta imagens/saída\n")
