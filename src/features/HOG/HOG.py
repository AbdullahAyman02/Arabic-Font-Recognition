import cv2
import numpy as np


def hog(image: np.ndarray) -> np.ndarray:
    '''
    Extracts the HOG features from the given image.

    Parameters:
    image (numpy.ndarray): The image from which the HOG features will be extracted.

    Returns:
    numpy.ndarray: The HOG features extracted from the image.
    '''
    # Create a HOG descriptor
    hog_descriptor = cv2.HOGDescriptor()

    # Compute the HOG features
    features = hog_descriptor.compute(image)

    return features
