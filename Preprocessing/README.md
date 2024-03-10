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
