# Development of a Deep Neural Convolutional Architecture for Classifying Diabetic Retinopathy Fundus Images

## Overview
Diabetic retinopathy is a complication of diabetes that affects the eyes, specifically the retina, and can lead to blindness, making early detection crucial. Based on this premise, the present thesis proposes a deep convolutional artificial neural network architecture for the classification of fundus photographs, identifying the presence or absence of diabetic retinopathy. This proposed network will be compared with a ResNet50 network to evaluate its performance, as the proposed network is intended to be computationally cost-effective and deployable on a laptop with internet connection.
Lastly, the integration of the developed architecture into a graphical interface will be explored.

## Status
For now, only the digital image preprocessing (preprocessing) and the ResNet model implemented in PyTorch Lightning are available in this repository.

**Note:** This project is currently a work in progress.

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

##Preprocessing
These data consist of fundus photographs, which were taken with different cameras, focusing, lighting, and qualities. This diversity highlights the importance of image preprocessing in the development of artificial intelligence systems for classification applications. Therefore, the first step in this preprocessing was to homogenize the data as much as possible.

### Region of Interest (ROI)
The images presented varied sizes, which required the first step to be resizing them. However, since the images were rectangular and contained a circle (the retina), distortion occurred, and the circle ended up becoming an oval. This situation is concerning because distortion can influence the characteristics of the disease in the retina.

To address this problem, a geometric algorithm was applied to crop the image to its region of interest (ROI). By having a square with the ROI, the resizing process no longer produces oval distortion.

The implemented algorithm first performs Otsu binarization, then obtains horizontal and vertical projections to extract the start and end of the retina, allowing the image to be cropped with a small margin. Zeros are also added in the missing section after cropping the image to obtain a square image.

### Graham-RGB
This preprocessing is named after a thesis from this institution (UPIITA), which is based on a report by Ben Graham from a Kaggle competition on diabetic retinopathy detection. The preprocessing can be presented as an equation proposed by Van Grinsven, based on Graham's report.

$I_{ce}(x, y; \sigma) = \alpha I(x, y) + \beta G(x, y; \sigma) * I(x, y) + \gamma$

Where:
- $I_{ce}(x, y; \sigma)$ is the enhanced image.
- $I(x, y)$ is the original image.
- $G(x, y; \sigma)$ is a Gaussian filter with scale $\sigma$.
- $*$ represents the convolution operator.
- $\alpha = 4$, $\beta = -4$, $\sigma = \frac{{\text{Tamaño de la imagen}}}{30}$, $\gamma = 128$

The parameters were empirically determined by Graham.

The preprocessing involves convolving the original image with a Gaussian filter, followed by a weighted pixel-wise addition with the original image using the values of the constants alpha, beta, and gamma. This blending allows extracting the local average color and thus reducing part of the variation between images due to different lighting conditions. 


