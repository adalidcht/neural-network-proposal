# -*- coding: utf-8 -*-
"""
@author: adalc
"""
#Clase de la red propuesta. Hereda de LightningModule
class RedPropuesta(LightningModule):
    #Inicializa la clase
    def __init__(self):
        #inicializa la clase LightningModule
        super().__init__()
        #Define las capas de la red
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16,64,3)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64,32,3)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3)
        self.conv4 = nn.Conv2d(32,50,3)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(5)
        self.conv5 = nn.Conv2d(50,116,5)
        self.relu5 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.conv6 = nn.Conv2d(116,30,6)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(30,12,3)
        self.relu7 = nn.ReLU()

        self.fc1 = nn.Linear(12*5*5, 300)
        self.fc2 = nn.Linear(300, 128)
        self.fc3 = nn.Linear(128, 2)


    def forward(self,x):
        x = self.conv1(x)
        X = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool3(x)
        x = self.conv5(x)
        x = self.drop1(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    #El resto del código es similar que el del resnet-50