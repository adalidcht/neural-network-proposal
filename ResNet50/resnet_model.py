# -*- coding: utf-8 -*-
"""
@author: adalc
"""

import torch
import torch.nn as nn
import albumentations as alb

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchmetrics import Accuracy
from torchvision.models import resnet50
from albumentations.pytorch.transforms import ToTensorV2

#Clase de la red ResNet50. Hereda de LightningModule
class ResNet50(LightningModule):

#Inicialización de la clase
    def __init__(self):
        #inicializa la clase LightningModule
        super().__init__()
        #Arquitectura del modelo. Resnet50
        self.model = resnet50(weights = 'DEFAULT')
        #Modifica la 'fully connected layer' para clasificación binaria
        self.model.fc = nn.Linear(2048, 2)
        #Entropía cruzada como función de pérdida
        self.loss_fn = nn.CrossEntropyLoss()
        #Inicializa el calculador de precisión para los datos de entrenamiento
        self.train_acc = Accuracy(task = "multiclass", num_classes = 2)
        #Inicializa el calculador de precisión para los datos de validación
        self.val_acc = Accuracy(task = "multiclass", num_classes = 2)
        #Inicializa el calculador de precisión para los datos de prueba
        self.test_acc = Accuracy(task = "multiclass", num_classes = 2)
        #Inicializa listas para almacenar las pérdidas
        #en los datos de entrenamiento, validación y prueba.
        self.train_losses = list()
        self.val_losses = list()
        self.test_losses = list()

#Propagación hacia delante del modelo ('forward pass')
    def forward(self, x):
        x = self.model(x)
        return x

#Define la iteración del entrenamiento
    def training_step(self, batch, batch_idx):
        #Desempaqueta el batch en values y labels
        values, labels = batch
        #Propagación hacia delante
        outputs = self(values)
        #Calcula la pérdida
        loss = self.loss_fn(outputs, labels)
        #Guarda la pérdida
        self.train_losses.append(loss.detach())
        #Calcula la precisión
        acc = self.train_acc(outputs, labels)
        #Registra la pérdida y la precisión en una barra de progreso
        self.log_dict({
            'train_loss': loss.detach(),
            'train_acc': acc
            }, prog_bar=True)

        #Regresa un diccionario con la pérdida
        return {'loss': loss}

#Define la iteración de validación similar al entrenamiento
    def validation_step(self, batch, batch_idx):

        values, labels = batch
        outputs = self(values)
        loss = self.loss_fn(outputs, labels)
        self.val_losses.append(loss)
        acc = self.val_acc(outputs, labels)
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc
            }, prog_bar=True)

        return {'loss': loss}

#Define la iteración de prueba similar al entrenamiento
    def test_step(self, batch, batch_idx):
        values, labels = batch
        outputs = self(values)
        loss = self.loss_fn(outputs, labels)
        self.test_losses.append(loss.detach())
        acc = self.test_acc(outputs, labels)
        self.log_dict({
            'test_loss': loss.detach(),
            'test_acc': acc
            }, prog_bar=True)

        return {'loss': loss}

#Realizan el promedio de pérdida y precisión, se muestra al  final de cada época
    def on_train_epoch_end(self) -> None:
        mean_loss = torch.stack(self.train_losses).mean()
        mean_acc = self.train_acc.compute()
        self.log_dict({
            'train_epoch_loss': mean_loss,
            'train_epoch_acc': mean_acc})
        self.train_losses.clear()

    def on_validation_epoch_end(self) -> None:
        mean_loss = torch.stack(self.val_losses).mean()
        mean_acc = self.val_acc.compute()
        self.log_dict({
            'val_epoch_loss': mean_loss,
            'val_epoch_acc': mean_acc})
        self.val_losses.clear()

    def on_test_epoch_end(self) -> None:
        mean_loss = torch.stack(self.test_losses).mean()
        mean_acc = self.test_acc.compute()
        self.log_dict({
            'test_epoch_loss': mean_loss,
            'test_epoch_acc': mean_acc})
        self.test_losses.clear()

#Configura el optimizador para el entrenamiento
    def configure_optimizers(self):
        #Optimizador Adam con una tasa de aprendizaje de 0.001
        optimizer = Adam(params = self.parameters(), lr=0.001)

        return optimizer


# Definen los cargadores de datos
    def train_dataloader(self):
        #Transformaciones de la imagen
        transforms = alb.Compose([
            alb.Normalize(),
            ToTensorV2()
        ])
        #Uso de la clase RetinaDataset
        #La clase debe de importarse, es el dataloader
        dataset = RetinaDataset(
            root='/.../train',
            transforms=transforms)
        #DataLoader para cargar los datos en batches
        loader = DataLoader(dataset = dataset,
                            batch_size = 10,
                            num_workers = 4,
                            shuffle = True)
        return loader

    def val_dataloader(self):
        transforms = alb.Compose([
            alb.Normalize(),
            ToTensorV2()
        ])
        dataset = RetinaDataset(
            root='/.../val',
            transforms=transforms)

        loader = DataLoader(dataset = dataset,
                            batch_size = 10,
                            num_workers = 4)
        return loader

    def test_dataloader(self):
        transforms = alb.Compose([
            alb.Normalize(),
            ToTensorV2()
        ])
        dataset = RetinaDataset(
            root='/.../test',
            transforms=transforms)

        loader = DataLoader(dataset = dataset,
                            batch_size = 10,
                            num_workers = 4)
        return loader