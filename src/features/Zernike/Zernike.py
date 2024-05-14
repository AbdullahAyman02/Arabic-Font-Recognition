'''
This module is responsible for extracting the Zernike moments from the images.
'''
import cv2
import mahotas
import numpy as np


def zernike(image: np.ndarray, degree: int = 8) -> np.ndarray:
    '''
    Extract Zernike moments from the given image.

    Parameters:
    image (numpy.ndarray): The image from which the Zernike moments will be extracted.
    degree (int): The degree of Zernike moments to be computed (default is 8).

    Returns:
    numpy.ndarray: The computed Zernike moments.
    '''
    # Convert image to grayscale if it is not already
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to a square
    size = max(image.shape)
    image = cv2.resize(image, (size, size))

    # Compute Zernike moments
    moments = mahotas.features.zernike_moments(image, degree)

    return moments
