import cv2
import matplotlib.pyplot as plt

imagePath = 'rosto.jpg'
img = cv2.imread(imagePath)
print(f'Dimensão Img original: {img.shape}')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f'Dimensão Escala de Cinza: {gray_image.shape}')
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_classifier.empty():
    print('Erro Classificador\n')
    exit()

face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()