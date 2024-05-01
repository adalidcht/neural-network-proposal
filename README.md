# Development of a Deep Neural Convolutional Architecture for Classifying Diabetic Retinopathy Fundus Images

## Overview
Diabetic retinopathy is a complication of diabetes that affects the eyes, specifically the retina, and can lead to blindness, making early detection crucial. Based on this premise, the present thesis proposes a deep convolutional artificial neural network architecture for the classification of fundus photographs, identifying the presence or absence of diabetic retinopathy. This proposed network will be compared with a ResNet50 network to evaluate its performance, as the proposed network is intended to be computationally cost-effective and deployable on a laptop with internet connection.
Lastly, the integration of the developed architecture into a graphical interface will be explored.

## Status
For now, only the digital image preprocessing (preprocessing) and the ResNet model implemented in PyTorch Lightning are available in this repository.

> [!Note]
> This project is currently a work in progress.

## Database Structuring
The images were gathered from the following databases:

- IDRiD (Indian Diabetic Retinopathy Image Dataset)
- EYEPACS 
- Messidor-2 
- APTOS-19 (Asia Pacific Tele-Ophthalmology Society) 

The dataset was ultimately reduced from 96,367 images to 4000 images. This reduction aimed to achieve homogeneity in the degrees of diabetic retinopathy across the various databases used for collection.

### Data Homogenization Process
This homogenization process focuses on standardizing data to ensure coherence and consistency throughout the database.

To homogenize the database, the characteristics of the images were established, as detailed in Table 4.4.

| Feature         | Description                                             |
|-----------------|---------------------------------------------------------|
| Format          | .png                                                    |
| Size            | 512x512 pixels                                         |
| Color Model     | RGB                                                     |

This setup allows the model to learn patterns and relationships present in the data during training, evaluate with unseen data during validation, and provide a final evaluation on completely new data during testing to obtain an objective measure of the model's capability.

## [Preprocessing](Preprocessing/README.md)
## [ResNet 50](ResNet50/README.md)
## [Neural Network Proposal](NN-Proposal/README.md)

