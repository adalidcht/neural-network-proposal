"""
@author: adalc
"""

import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

#Clase de obtención de la base de datos. Hereda de Dataset
class RetinaDataset(Dataset):

    def __init__(self, root, transforms) -> None:
        #Asegurar que se ejecute la clase Dataset
        super().__init__()
        #Atributos del método __init__
        #Transformaciones
        #Rutas de las imágenes
        #Conjunto de etiquetas
        self.transforms = transforms
        self.images = list()
        label = set()

        #Recorre el directorio de las imagenes y almacena las etiquetas
        for root, _, files in os.walk(root):
            for filename in files:
                path = os.path.join(root, filename)
                lbl = root.split(os.sep)[-1]
                label.add(lbl)
                #Guarda las rutas de las imágenes
                self.images.append(path)
        #Acomodo de etiquetas
        self.labels = dict(enumerate(label))
        self.labels = {v: k for k, v in self.labels.items()}

    #Obtener las imágenes
    def __getitem__(self, index):
        #A la imagen seleccionada se obtiene su etiqueta correspondiente
        name = self.images[index]
        key = name.split(os.sep)[-2]
        label = self.labels[key]
        #Abrir la imagen y convertirla en un array
        image = Image.open(name)
        image = np.array(image)
        #Transformar el array en tensor
        tensor = self.transforms(image=image)['image']

        #Convertir la etiqueta en un tensor de tipo long
        label = torch.tensor(label).long()

        #Retorno de tupla imagen y etiqueta
        return tensor, label

    #Longitud del conjunto de datos
    def __len__(self):
        return len(self.images)