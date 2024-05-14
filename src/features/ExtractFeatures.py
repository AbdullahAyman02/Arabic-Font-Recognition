from GaborFilter.GaborFilter import get_gabor_vect
from EnclosingArea.EnclosingArea import enclosing_area
from GLCM.GLCM import glcm_features
from LBP.LBP import extract_lbp_features

import numpy as np
def extract_features(image,kernels):
    area = enclosing_area(image)
    # convert area to a 1D array
    area = area.flatten()
    gabor = get_gabor_vect(image,  kernels)
    glcm = glcm_features(image)
    lbp = extract_lbp_features(image)
    # convert lbp to a 1D array
    lbp = lbp.flatten()
    features = np.concatenate((area, gabor, glcm, lbp))
    return features
