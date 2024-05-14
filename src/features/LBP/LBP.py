import cv2
import numpy as np
from skimage import feature


# TODO: rename this to lbp

def extract_lbp_features(image: np.ndarray, num_points: int = 24, radius: int = 8, eps: float = 1e-7) -> np.ndarray:
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
