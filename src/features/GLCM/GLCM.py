import cv2
import numpy as np
from skimage import feature
from typing import List, Tuple


def glcm_features(image: np.ndarray, distances: List[int] = [5], angles: List[float] = [0], levels: int = 256, symmetric: bool = True, normed: bool = True) -> np.ndarray:
    '''
    Extracts GLCM texture features from the given image.

    Parameters:
    image (numpy.ndarray): The input image.
    distances (List[int]): List of pixel pair distances. Default is [5].
    angles (List[float]): List of angles in radians. Default is [0].
    levels (int): Number of gray levels. Default is 256.
    symmetric (bool): Whether to make the GLCM matrix symmetric. Default is True.
    normed (bool): Whether to normalize the GLCM matrix. Default is True.

    Returns:
    numpy.ndarray: Array of texture features.
    '''
    # Convert image to grayscale if it is not already
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute GLCM
    glcm = feature.greycomatrix(
        image, distances, angles, levels, symmetric=symmetric, normed=normed)

    # Compute texture features
    properties = ['contrast', 'dissimilarity',
                  'homogeneity', 'energy', 'correlation']
    features = [feature.greycoprops(glcm, prop)[0, 0] for prop in properties]

    return np.array(features)
