# Trabalho 3 - MC920
# Gustavo Salibi (174135)

import cv2
import numpy as np
from skimage.morphology import binary_dilation, binary_erosion, binary_closing

def conta_transicoes(img, min_x, min_y, largura, altura):
    img_altura, img_largura = img.shape
    transicoes = 0
    for j in range(altura):
        for i in range(largura):
            x = min_x + i
            y = min_y + j
            if x < img_largura and y < img_altura and img[y,x]:
                if x - 1 >= min_x and not img[y, x - 1]:
                    transicoes += 1
                if y - 1 >= min_y and not img[y - 1, x]:
                    transicoes += 1
    return transicoes


def cria_mascara(estatisticas, shape):
    mascara = np.zeros(shape)
    for item in estatisticas:
        min_x, min_y, largura, altura, temp = item
        mascara[min_y:min_y+altura, min_x:min_x+largura] = 1
    return mascara.astype(np.bool)


def desenha_saida(img, estatisticas, saida_8=False):   
    i = 0
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for temp, item in enumerate(estatisticas):
        i += 1
        min_x, min_y, w, h, area = item
        cv2.rectangle(img, (min_x, min_y), (min_x+w, min_y+h), (0, 0, 255), 3)
        if saida_8:
            cv2.putText(img, '%d' % (i), (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    return img


# abre imagem
img_original = cv2.imread('imagens/entrada/imagem_entrada.pbm', cv2.IMREAD_GRAYSCALE)
img = ~((img_original / 255).astype(np.bool))

# 1. dilatação da imagem original com um elemento estruturante de 1 pixel de altura e 100 pixels de largura
saida1 = binary_dilation(img, selem=np.ones((1, 100)))

# 2. erosão da imagem resultante com o mesmo elemento estruturante do passo (1)
saida1 = binary_erosion(saida1, selem=np.ones((1, 100)))

# 3. dilatação da imagem original com um elemento estruturante de 200 pixels de altura e 1 pixel de largura
saida2 = binary_dilation(img, selem=np.ones((200, 1)))

# 4. erosão da imagem resultante com o mesmo elemento estruturante do passo (3)
saida2 = binary_erosion(saida2, selem=np.ones((200, 1)))

# 5. aplicação da intersecção (AND) dos resultados dos passos (2) e (4)
img = saida1 & saida2

# 6. fechamento do resultado obtido no passo (5) com um elemento estruturante de 1 pixel de altura e 30 pixels de largura
img = binary_closing(img, selem=np.ones((1,30)))

# 7. aplicação de algoritmo para identificação de componentes conexos sobre o resultado do passo (6)
componentes_estatisticas = cv2.connectedComponentsWithStats(255*img.astype(np.uint8), 4, cv2.CV_32S)[2][1:]

# 8. para cada retângulo envolvendo um objeto, calcule:
# 8a. razão entre o número de pixels pretos e o número total de pixels (altura × largura)
# 8b. razão entre o número de transições verticais e horizontais branco para preto e o número total de pixels pretos
razoes = []
i = 0
for item in componentes_estatisticas:
    i += 1
    min_x, min_y, largura, altura, pretos = item
    razao_pixels = pretos / (largura*altura) # 8a
    transicoes = conta_transicoes(img, min_x, min_y, largura, altura) # 8b
    razao_transicoes = transicoes / pretos # 8b
    razoes.append((razao_pixels, razao_transicoes))
    print("Retângulo {:>2}: pretos/total={:.4f} | transições/pretos={:.4f}".format(i, razao_pixels, razao_transicoes))
saida_8 = desenha_saida(((~img) * 255).astype(np.uint8), componentes_estatisticas, True)
cv2.imwrite('imagens/saída/saida_8.png', saida_8)

# 9. criação de uma regra para classificar cada componente conexo, de acordo com as medidas obtidas no passo (8), como texto e não texto
texto_estatisticas = []
for item, stat in enumerate(componentes_estatisticas):
    razao_pixels, razao_transicoes = razoes[item]
    if (.5, .9)[0] < razao_pixels < (.5, .9)[1] \
            and (0, .1)[0] < razao_transicoes < (0, .1)[1]:
        texto_estatisticas.append(stat)
saida_9 = desenha_saida(img_original, texto_estatisticas)
cv2.imwrite('imagens/saída/saida_9.png', saida_9)

# 10. aplicação de operadores morfológicos apropriados para segmentar cada linha do texto em blocos de palavras. Coloque um retângulo 
#envolvendo cada palavra na imagem original. Calcule o número total de linhas de texto e de blocos de palavras na imagem
palavras = binary_dilation(~((img_original / 255).astype(np.bool)), selem=np.ones((6, 1)))
palavras = binary_closing(palavras, selem=np.ones((1, 10))) # segmenta cada linha em blocos de palavras
cv2.imwrite('imagens/saída/saida_10-blocos-palavras.png', ((~palavras) * 255).astype(np.uint8))  # segmenta cada linha em blocos de palavras

palavras_estatisticas = cv2.connectedComponentsWithStats(255*palavras.astype(np.uint8), 4, cv2.CV_32S)[2][1:]                                                                                         # 10
palavras_mascara = cria_mascara(palavras_estatisticas, img.shape) & cria_mascara(texto_estatisticas, img.shape)
palavras_estatisticas = cv2.connectedComponentsWithStats(255*palavras_mascara.astype(np.uint8), 4, cv2.CV_32S)[2][1:]                                                                                    # 10
palavras_com_ret = desenha_saida(img_original, palavras_estatisticas)
cv2.imwrite('imagens/saída/saida_10-palavras-com-retangulo.png', palavras_com_ret) # palavras com retângulos

#  imprime número total de linhas de texto e de blocos de palavras na imagem
print('Total de linhas: {}\nTotal de blocos de palavras: {}'.format(len(texto_estatisticas), len(palavras_estatisticas)))
