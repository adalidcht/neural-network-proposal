# -*- coding: utf-8 -*-
"""
@author: adalc
"""

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


#Se ejecuta como un script principal
if __name__ == '__main__':
    model = RedPropuesta()


    logger = TensorBoardLogger(save_dir = checkpoint_dir, name = 'log')

    #Se especifíca las epocas y se activa el checkpoint
    trainer = Trainer(max_epochs = 50,
                      callbacks = [ckpt_callback],
                      logger=logger, log_every_n_steps=10)

    #Realiza el entrenamientousando el método fit
    trainer.fit(model = model)
    trainer.validate()
    trainer.test()