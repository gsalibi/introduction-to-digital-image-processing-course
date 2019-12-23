from sklearn.cluster import KMeans, MiniBatchKMeans
import cv2
import numpy as np

# imprime menu
print('''
Selecione a imagem a ser trabalhada:
    1 - baboon.png
    2 - monalisa.png
    3 - peppers.png
    4 - watch.png
        ''')

# seleciona opção
op = 0
while op == 0:
    op = int(input())
    if op == 1:
        img_dir = "imagens/entrada/baboon.png"
        img_nome = "baboon"
    elif op == 2:
        img_dir = "imagens/entrada/monalisa.png"
        img_nome = "monalisa"
    elif op == 3:
        img_dir = "imagens/entrada/peppers.png"
        img_nome = "peppers"
    elif op == 4:
        img_dir = "imagens/entrada/watch.png"
        img_nome = "watch"
    else:
        print("Opção inválida. Tente novamente.")
        op = 0
print('Quantidade de cores: ')
n_cores = int(input())
print("processando imagens...\n")

# 1 - ler a imagem colorida de entrada
img = cv2.imread(img_dir).astype(np.float32)

# 2 - aplicar a técnica k-means de agrupamento de dados para encontrar grupos de cores mais representativas
# 3 - salvar dicionário (codebook) gerado pela técnica de agrupamento, ou seja, os centros dos grupos e os rótulos correspondentes a cada pixel da imagem
altura, largura, canais = img.shape
cores = img.reshape((altura*largura, canais))
codebook = MiniBatchKMeans(n_cores).fit(cores)

# 4 - reconstruir a imagem com cores reduzidas a partir do dicionário armazenado
saida = np.zeros((codebook.labels_.size, 3))
for label, cor in enumerate(codebook.cluster_centers_):
    saida[codebook.labels_ == label] = cor
saida = saida.reshape(img.shape).astype(np.uint8)

cv2.imwrite('imagens/saída/{}-{}.png'.format(img_nome,n_cores), saida)

print("finalizado\n")