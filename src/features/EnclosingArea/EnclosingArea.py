import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, io, measure, morphology, util
from skimage.feature import local_binary_pattern
from typing import Union


def get_area_ratio(image: np.ndarray) -> float:
    '''
    Computes the ratio of the sum of area of closed characters to the area of the image.

    Parameters:
    image (numpy.ndarray): Input image.

    Returns:
    float: Ratio of the sum of area of closed characters to the area of the image.
    '''
    # Remove small holes
    mask = morphology.remove_small_holes(image, area_threshold=10)
    masked = np.where(mask, image, 255)
    num_black_pixels = np.sum(masked == 0)
    total_area = masked.shape[0] * masked.shape[1]
    ratio = num_black_pixels / total_area
    return ratio


def plot_1D_feature(features: pd.DataFrame) -> None:
    '''
    Plots 1D feature values for different classes.

    Parameters:
    features (pd.DataFrame): DataFrame containing features and class labels.
    '''
    classes = features['font_type'].unique()
    colors = ['r', 'g', 'b', 'y']  # Color map for different classes

    for i, cls in enumerate(classes):
        feature_values = features[features['font_type'] == cls]['area_ratio']
        x = np.arange(len(feature_values))
        plt.scatter(x, feature_values, color=colors[i], label=cls)

    plt.legend()
    plt.show()


def enclosing_area(image: np.ndarray) -> float:
    '''
    Computes the ratio of the sum of area of closed characters to the area of the image.

    Parameters:
    image (numpy.ndarray): Input image.

    Returns:
    float: Ratio of the sum of area of closed characters to the area of the image.
    '''
    return get_area_ratio(image)
