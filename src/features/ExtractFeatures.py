from GaborFilter.GaborFilter import get_gabor_vect, generate_gabor_kernel_ski
from EnclosingArea.EnclosingArea import enclosing_area
from GLCM.GLCM import glcm_features
from LBP.LBP import extract_lbp_features
from sklearn.preprocessing import StandardScaler
import numpy as np


def extract_features(image, kernels):
    '''
    Extract all features from the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernels (list): List of Gabor kernels.

    Returns:
        numpy.ndarray: Concatenated feature vector.
    '''
    area = enclosing_area(image)
    # Convert area to a 1D array
    area = area.flatten()
    gabor = get_gabor_vect(image, kernels)
    glcm = glcm_features(image)
    lbp = extract_lbp_features(image)
    # Convert lbp to a 1D array
    lbp = lbp.flatten()
    features = np.concatenate((area, gabor, glcm, lbp))
    return features


def normalize(x_data):
    '''
    Normalize the input data.

    Parameters:
        x_data (numpy.ndarray): Input data to be normalized.

    Returns:
        StandardScaler: StandardScaler instance fitted to the input data.
    '''
    scaler = StandardScaler()
    scaler.fit(x_data)
    return scaler
