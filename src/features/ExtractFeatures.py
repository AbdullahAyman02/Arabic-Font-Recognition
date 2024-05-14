from GaborFilter.GaborFilter import get_gabor_vect, generate_gabor_kernel_cv
from EnclosingArea.EnclosingArea import enclosing_area
import numpy as np
def extract_features(image,kernels):
    area = enclosing_area(image)
    gabor = get_gabor_vect(image,  kernels)
    return np.concatenate([[area] , gabor])
