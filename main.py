import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

weights_path = "lib/yolov3.weights"
config_path = "lib/yolov3.cfg"
classes_path = "lib/coco.names"  # Arquivo de nomes das classes

# Verificar se os arquivos necessários existem
if not os.path.exists(weights_path):
    print(f"Erro: Arquivo de pesos não encontrado em {weights_path}")
    exit()

if not os.path.exists(config_path):
    print(f"Erro: Arquivo de configuração não encontrado em {config_path}")
    exit()

if not os.path.exists(classes_path):
    print(f"Erro: Arquivo de nomes das classes não encontrado em {classes_path}")
    exit()

# Carregar os nomes das classes
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Carregar os arquivos da rede neural YOLO pré-treinada
try:
    net = cv2.dnn.readNet(weights_path, config_path)
except cv2.error as e:
    print(f"Erro ao carregar a rede YOLO: {e}")
    exit()

# Obter os nomes das camadas de saída
layer_names = net.getLayerNames()

# Ajustar o índice para versões diferentes do OpenCV
try:
    # Para versões mais recentes do OpenCV (> 4.0)
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    # Para versões mais antigas do OpenCV
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Carregar a imagem
imagePath = 'img/corpo4.jpg'
img = cv2.imread(imagePath)

if img is None:
    print(f"Erro ao carregar a imagem em {imagePath}")
    exit()

height, width, channels = img.shape

# Pré-processamento da imagem para YOLO
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Listas para armazenar as informações das detecções
class_ids = []
confidences = []
boxes = []

# Analisar as detecções
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filtrar detecções com confiança acima de um limite
        if confidence > 0.5:
            # Coordenadas da caixa delimitadora
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Coordenadas da caixa delimitadora
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Armazenar as informações das detecções
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Aplicar supressão não máxima para eliminar caixas delimitadoras sobrepostas
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Desenhar as caixas delimitadoras e rótulos na imagem
if len(indices) > 0:
    for i in indices.flatten():  # Ajustado para usar .flatten() para versões mais recentes
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (255, 0, 255)  # Verde
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
        cv2.putText(img, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Exibir a imagem usando matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
