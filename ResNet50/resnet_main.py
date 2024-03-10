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
    #Guarda los puntos de control del modelo durante el entrenamiento
    '''
    filename = nombre del archivo
    Cuando la precisión en los datos de validación se máxima se guarda:
    -monitor, mode
    save_last = guarda el último modelo entrenado
    save_weights_only = guarda solo los pesos no la arquitectura
    verbose = Muestra los puntos de control
    '''
    ckpt_callback = ModelCheckpoint(
        filename = 'model',
        monitor = 'val_epoch_acc',
        mode = 'max',
        save_last = True,
        save_weights_only = True,
        verbose = True
        )
    #Se especifíca las epocas y se activa el checkpoint
    trainer = Trainer(max_epochs = 25,callbacks = [ckpt_callback])
    #Realiza el entrenamientousando el método fit
    trainer.fit(model = model)
    #Imprime la ruta al mejor modelo guardado según los puntos de control
    print(ckpt_callback.best_model_path)
    #Evalua el modelo con el conjunto de prueba
    trainer.test()

    #Carga el modelo desde el último punto de control guardado
    ResNet50.load_from_checkpoint(checkpoint_path = '*.ckpt')
