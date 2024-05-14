import cv2
import numpy as np
from skimage import feature


# TODO: rename this to lbp

def extract_lbp_features(image: np.ndarray, num_points: int = 24, radius: int = 8, eps: float = 1e-7) -> np.ndarray:
    '''
    Extract Local Binary Pattern (LBP) features from the given image.

    Parameters:
    image (numpy.ndarray): The image from which the LBP features will be extracted.
    num_points (int): Number of points to sample circularly around each pixel (default is 24).
    radius (int): Radius of circle around each pixel (default is 8).
    eps (float): Small value added to avoid division by zero when normalizing the histogram (default is 1e-7).

    Returns:
    numpy.ndarray: The normalized histogram of LBP features.
    '''
    # Compute the LBP representation of the image
    lbp = feature.local_binary_pattern(
        image, num_points, radius, method="uniform")

    # Compute the histogram of the LBP
    hist = cv2.calcHist([lbp.astype('float32')], [0], None, [
                        num_points + 2], [0, num_points + 2])

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    return hist
