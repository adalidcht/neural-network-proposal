# Preprocessing: Digital Image Processing

## Overview
These techniques are essential for preparing fundus images before feeding them into the neural network for classification. The preprocessing steps aim to enhance image quality, remove noise, and extract relevant features to improve the performance of the classification model.

## Techniques
- **Region of Interest (ROI) Extraction**: Identifying and extracting regions of interest to focus on specific areas of the image.
- **Graham-RGB Enhancement**: Applying Graham-RGB algorithm to improve image contrast and illumination.
- **Normalization**: Normalizing pixel values to a standard range for better convergence during training.

## Implementation
The preprocessing techniques mentioned above are implemented using Python libraries such as OpenCV, NumPy and Albumentations. 

## Usage
To utilize the preprocessing module:
1. Ensure you have the required dependencies installed (OpenCV, NumPy, Albumentations).
2. Import the preprocessing functions from the provided module.
3. Apply the desired preprocessing techniques to your fundus images before feeding them into the classification model.

## Preprocessing
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
