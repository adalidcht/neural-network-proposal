# ResNet 50
The development of the ResNet 50 model was carried out in **Google Colaboratory**, utilizing the **PyTorch Lightning** framework, thereby employing the **Object-Oriented Programming** (OOP) paradigm.

## Object-Oriented Programming (OOP) Paradigm
The object-oriented programming paradigm facilitates the management and maintenance of code more easily. Among its advantages is the reusability of objects in different parts of the code, allowing for class inheritance to create variations of the same model. It is easy to understand when implementing classes and their methods since one simply calls the method of interest without needing to know the details of the method's algorithm. Lastly, the code is easily scalable if there is a need to increase the layers of the network, expand the dataset, or experiment with different hyperparameter configurations.

## Implementation Structure
The implementation of the network is divided into 3 scripts:
- **Data Loader**: Responsible for loading and preprocessing the dataset.
- **Network Model**: Contains the implementation of the ResNet 50 architecture.
- **Main Implementation**: The main script where the training and evaluation procedures are conducted.

**Note:** The use of PyTorch Lightning framework simplifies the training process and provides useful abstractions for handling data loading, model training, and evaluation.

## Usage
To utilize the ResNet 50 model:
1. Set up the necessary environment with PyTorch and PyTorch Lightning installed.
2. Import the required scripts/modules.
3. Initialize the Data Loader and Model.
4. Execute the Main script to start the training and evaluation processes.

