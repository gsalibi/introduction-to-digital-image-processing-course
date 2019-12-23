import numpy as np
import cv2

# imprime menu
print('''
Selecione a imagem a ser trabalhada:
    1 - Foto1
    2 - Foto2
    3 - Foto3
    4 - Foto4
    5 - Foto5
        ''')

# seleciona opção
op = 0
while op == 0:
    op = int(input())
    if op == 1:
        imgA_caminho = "imagens/entrada/foto1A.jpg"
        imgB_caminho = "imagens/entrada/foto1B.jpg"
        img_nome = "foto1"
    elif op == 2:
        imgA_caminho = "imagens/entrada/foto2A.jpg"
        imgB_caminho = "imagens/entrada/foto2B.jpg"
        img_nome = "foto2"
    elif op == 3:
        imgA_caminho = "imagens/entrada/foto3A.jpg"
        imgB_caminho = "imagens/entrada/foto3B.jpg"
        img_nome = "foto3"
    elif op == 4:
        imgA_caminho = "imagens/entrada/foto4A.jpg"
        imgB_caminho = "imagens/entrada/foto4B.jpg"
        img_nome = "foto4"
    elif op == 5:
        imgA_caminho = "imagens/entrada/foto5A.jpg"
        imgB_caminho = "imagens/entrada/foto5B.jpg"
        img_nome = "foto5"
    else:
        print("Opção inválida. Tente novamente.")
        op = 0
print("processando imagens...\n")


# abre imagens coloridas
imgA_pb_cor = cv2.imread(imgA_caminho, cv2.IMREAD_COLOR)
imgB_pb_cor = cv2.imread(imgB_caminho, cv2.IMREAD_COLOR)

# 1 - converter as imagens coloridas de entrada em imagens de níveis de cinza
imgA_pb = cv2.imread(imgA_caminho, cv2.IMREAD_GRAYSCALE)
imgB_pb = cv2.imread(imgB_caminho, cv2.IMREAD_GRAYSCALE)

# 2 - encontrar pontos de interesse e descritores invariantes locais para o par de imagens
descritores = ['-sift', '-surf', '-orb', '-brief']
componentes = [cv2.xfeatures2d.SIFT_create().detectAndCompute(imgA_pb, None),
               cv2.xfeatures2d.SIFT_create().detectAndCompute(imgB_pb, None),
               cv2.xfeatures2d.SURF_create().detectAndCompute(imgA_pb, None),
               cv2.xfeatures2d.SURF_create().detectAndCompute(imgB_pb, None),
               cv2.ORB_create().detectAndCompute(imgA_pb, None),
               cv2.ORB_create().detectAndCompute(imgB_pb, None),
               cv2.xfeatures2d.BriefDescriptorExtractor_create().compute(imgA_pb, cv2.xfeatures2d.StarDetector_create().detect(imgA_pb, None)),
               cv2.xfeatures2d.BriefDescriptorExtractor_create().compute(imgB_pb, cv2.xfeatures2d.StarDetector_create().detect(imgB_pb, None))]

for i in range(4):
    kpA, desA = componentes[2*i]
    kpB, desB = componentes[2*i + 1]

    # 3 - computar distâncias (similaridades) entre cada descritor das duas imagens
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) if descritores[i] == '-sift' or descritores[i] == '-surf' else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(desA, desB), key=lambda x:x.distance)

    # 4 - selecionar as melhores correspondências para cada descritor de imagem
    melhores = matches[:20]

    # 5 - executar a técnica RANSAC (RANdom SAmple Consensus) para estimar a matriz de homografia (cv2.findHomography)
    if len(matches) < 4:
        print('Quantidade de keypoints insuficiente.')
    ptsA = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

    # 6 - aplicar uma projeção de perspectiva (cv2.warpPerspective) para alinhar as imagens
    img_perspectiva = cv2.warpPerspective(imgA_pb_cor, H, (imgA_pb_cor.shape[1] + imgB_pb_cor.shape[1], max(imgA_pb_cor.shape[0], imgB_pb_cor.shape[0])))
    img_perspectiva[0:imgB_pb.shape[0], 0:imgB_pb.shape[1]] = imgB_pb_cor

    # 7 - unir as imagens alinhadas e criar a imagem panorâmica
    cv2.imwrite('imagens/saída/' + img_nome + descritores[i] + '.jpeg', img_perspectiva)

    # 8 - desenhar retas entre pontos correspondentes no par de imagens
    cv2.imwrite('imagens/saída/' + img_nome + descritores[i] + '-matches.jpeg', cv2.drawMatches(imgA_pb, kpA, imgB_pb, kpB, melhores, None, flags=2))

print("finalizado\n")