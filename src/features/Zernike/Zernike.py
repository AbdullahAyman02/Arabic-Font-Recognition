'''
This module extracts features from an image using Zernike moments.
'''

import numpy as np
import cv2
import mahotas

def zernike(image, degree=8):
    # Convert image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to a square to ensure moments are computed correctly
    (h, w) = image.shape
    if h > w:
        image = cv2.resize(image, (h, h))
    else:
        image = cv2.resize(image, (w, w))

    # Compute Zernike moments
    moments = mahotas.features.zernike_moments(image, degree)

    return moments