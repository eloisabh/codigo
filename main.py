import cv2
import matplotlib.pyplot as plt

imagePath = 'corpo2.jpeg' #indicando o caminho da imagem
img = cv2.imread(imagePath) #lê a imagem com a função .imread() do OpenCV
#Isso carregará a imagem do caminho do arquivo especificado e a retornará na forma de uma matriz Numpy

'''Opta-se pela utilização de matriz Numpy principalmente por conta da performance desta. Normalmente
    utilizada por conta da alta capacidades de manipulação e operação com grandes volumes de dados numéricos.
    Observa-se que todos os elementos devem ser do mesmo tipo (por exemplo, todos inteiros ou todos de ponto flutuante).
    É possível realizar operações vetorizadas, ou seja, aplicar operações matemáticas a todos os elementos de uma vez, 
    sem a necessidade de laços explícitos (loops)'''

if img is None:
    print("Erro ao carregar imagem!")
    exit()

print(f'Shape Img Colorida: {img.shape}') #Shape retorna uma tupla que contém as dimensões da imagem (largura, altura, qtd canais de cores)

#UTILIZAÇÃO DO HAAR CASCADE (CLASSIFICADOR)
'''Algoritmo popular usado para detecção de objetos em imagens, especialmente na detecção de rostos
    Padrões retangulares são utilizados para identificar contrastes entre áreas claras e escuras de uma imagem. 
    Essas características são aplicadas sobre uma imagem e, em seguida, um classificador treina sobre estas 
    características para detectar padrões específicos'''

#Representação de cores de imagem não é RGB e sim BGR

#Para aumentar a eficiência computacional, converte-se a imagem em escala de cinza antes de executar a detecção de faces
grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f'Shape Img P&B {grey_image.shape}')
cv2.imshow('P&B', grey_image)
cv2.waitKey(0)

#Carregando o classificador pré-treinado integrado ao OpenCV
#haarcascade_fullbody.xml (detectar corpo inteiro na entrada visual)
body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

#Verifica se o classificador foi carregado corretamente
if body_cascade.empty():
    print('Erro ao carregar o classificador!')
    exit()

#Executar a detecção de pessoas na imagem em escala de cinza
body = body_cascade.detectMultiScale(grey_image, scaleFactor=1.1, minNeighbors=3)
    #detectMultiScale(): metodo utilizado para identificar se há o padrão que procuramos
        #grey_image: imagem na escala de cinza criada anteriormente
        #scaleFactor: reduz o tamanho da imagem de entrada (1.1 - reduz 10%)
        #minNeighbors: especifica quantas regiões sobrepostas precisam existir para que o algoritmo considere
#           essa detecção confiável
        #minSize: define o tamanho do objeto a ser identificado

for (x, y, w, h) in body:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #(0, 255, 0) é a cor da caixa delimitadora
        # 4 indica a espessura

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converte de BGR para RGB

plt.figure(figsize=(40,20))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()