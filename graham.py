#Bibliotecas importadas
import os
import cv2

from tqdm import tqdm
from skimage.io import imread

import matplotlib.pyplot as plt
#-----------------------
#Funciones

def graham_rgb(image,alpha,beta,gamma,aspect):
    #Filtro gaussiano
    blured = cv2.GaussianBlur(image,(0,0),aspect/30)
    plt.figure()
    plt.imshow(blured)
    #Ecuación
    cvgraham = cv2.addWeighted(image,alpha,blured,beta,gamma)
    plt.figure()
    plt.imshow(cvgraham)
    return cvgraham

#Parámetros
alpha = 4
beta = -4
gamma = 128
aspect = 512

plt.close('all')

#Se carga el path de las imágenes
image = imread('E:\\TT\\Data\\preprocess\\prueba10.jpg')
plt.figure()
plt.imshow(image)
imagen_procesada = graham_rgb(image,alpha,beta,gamma,aspect)




# path_list = os.listdir(path)
# data_list = sorted(path_list)

# #Ciclo for para preprocesar cada una de las imágenes
# for imagen in tqdm(data_list, desc='Procesando imágenes', unit='imagen'):
#     ruta_imagen = os.path.join(path, imagen)
#     image = imread(ruta_imagen)
    
#     # Preprocesar la imagen
#     imagen_procesada = graham_rgb(image,alpha,beta,gamma,aspect)
    

