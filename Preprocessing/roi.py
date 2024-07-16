# -*- coding: utf-8 -*-
"""
@author: adalc
"""

#Bibliotecas importadas
import numpy as np

from skimage import color
from skimage import filters
from skimage.io import imread
from skimage.transform import resize

#----------------------
#Funciones

#Función para binarizar la image\Staten
def rgb2binary(image):
    gray = color.rgb2gray(image)
    umbral =  filters.threshold_otsu(gray)
    binary= gray > umbral
    return binary

#Función para obtener los límites de la retina
def get_limits(proj, margin, limit_threshold):
    lower_limit = None
    upper_limit = None
    #Obtiene el cambio en la proyección
    for i in range(1, len(proj) - 1):
        if proj[i - 1] < proj[i] or proj[i] > proj[i + 1]:
            if lower_limit is None:
                lower_limit = i
            upper_limit = i
    
    #Coloca un margen para tener pixeles alrededor de la retina
    if lower_limit > limit_threshold:
        lower_limit -= margin
    if upper_limit < len(proj) - (limit_threshold):
        upper_limit += margin
    
    return lower_limit, upper_limit

#Función para obtener la región de interés
def roi(image, new_width,new_height):
    image = image[:,:,:3]
    binary = rgb2binary(image)
    
    #perfiles o proyecciones en los ejes
    h_proj = np.sum(binary,axis = 0)
    v_proj = np.sum(binary,axis = 1)
    
    margin = 25
    limit = 50
    lower_limit_h,upper_limit_h = get_limits(h_proj,margin,limit)
    lower_limit_v,upper_limit_v = get_limits(v_proj,margin,limit)
    
    #Se recorta la región de interés
    cropped_image = image[lower_limit_v:upper_limit_v,
                          lower_limit_h:upper_limit_h]
    
    #Se crea una imagen en ceros para incrementar
    #la sección de la imagen faltante y hacer un cuadrado
    max_dimension = max(cropped_image.shape[0],cropped_image.shape[1])
    square_image = np.zeros((max_dimension,max_dimension,3), dtype=np.uint8)
    
    #Dónde iniciar la posición de la imagen en el fondo negro
    if cropped_image.shape[0] < cropped_image.shape[1]:
        start = int((max_dimension - cropped_image.shape[0]) / 2)
    else:
        start = 0
    
    #Se agrega la sección en ceros
    square_image[start:start + cropped_image.shape[0],
                 :cropped_image.shape[1]] = cropped_image
    
    resized_image = resize(square_image,
                           (new_width, new_height),
                           mode='constant',
                           preserve_range=True)
    resized_image = np.asarray(resized_image, dtype=np.uint8)
    
    return resized_image

#----------------------
#Código principal

#Tamaño del resize
new_width,new_height = 512,512 
#Uso de la función roi
new_image = roi(image,new_width,new_height)