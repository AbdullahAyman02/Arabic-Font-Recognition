'''
This module is responsible for extracting the HOG (Histogram of Oriented Gradients) features from the images.
'''

import cv2

def hog(image):
    '''
    Extracts the HOG features from the given image.

    Parameters:
    image (numpy.ndarray): The image from which the HOG features will be extracted.

    Returns:
    numpy.ndarray: The HOG features extracted from the image.
    '''
    # Create a HOG descriptor
    hog = cv2.HOGDescriptor()

    # Compute the HOG features
    features = hog.compute(image)

    return features
