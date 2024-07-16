# -*- coding: utf-8 -*-
"""
@author: adalc
"""

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

#Se ejecuta como un script principal
if __name__ == '__main__':
    #Instancia del modelo ResNet50
    #Resnet es una clase que debe importarse
    model = ResNet50()

    #Se especifíca las epocas y se activa el checkpoint
    trainer = Trainer(max_epochs = 25,callbacks = [ckpt_callback])
    #Realiza el entrenamientousando el método fit
    trainer.fit(model = model)
    #Evalua el modelo con el conjunto de prueba
    trainer.test()
