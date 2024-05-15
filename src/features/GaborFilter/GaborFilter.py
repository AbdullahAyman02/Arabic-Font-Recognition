import cv2
import numpy as np
import skimage as ski
from skimage import io, morphology, transform, filters, img_as_ubyte


def resize_image(image: np.ndarray, width: float, height: float) -> np.ndarray:
    '''
    Resize the input image to the specified width and height.

    Parameters:
        image (np.ndarray): Input image.
        width (float): Target width.
        height (float): Target height.

    Returns:
        np.ndarray: Resized image.
    '''
    return transform.resize(image, (width, height))


def get_frequencies(width: int, height: int) -> np.ndarray:
    '''
    Generate frequencies for Gabor filters based on the image dimensions.

    Parameters:
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        np.ndarray: Array of frequencies for Gabor filters.
    '''
    max_freq = width / 4
    frequencies = np.linspace(1, max_freq, num=5)
    return frequencies[1:]


def convert_to_grayscale(binary_image: np.ndarray) -> np.ndarray:
    '''
    Convert a binary image to grayscale. 

    Parameters:
        binary_image (np.ndarray): Binary image.

    Returns:
        np.ndarray: Grayscale image.
    '''
    grayscale_image = img_as_ubyte(binary_image)
    return grayscale_image


def apply_gabor_filters_ski(image: np.ndarray, thetas: list[float], frequencies: list[float]) -> list:
    '''
    Find Gabor filters using skimage then apply them and compute statistical features.

    Parameters:
        image (np.ndarray): Input image.
        thetas (list[float]): List of orientations for Gabor filters.
        frequencies (list[float]): List of frequencies for Gabor filters.

    Returns:
        list: List of statistical features for each filter.
    '''
    features = []
    for theta in thetas:
        for frequency in frequencies:
            filtered_real, _ = filters.gabor(
                image, frequency=frequency, theta=theta)
            mean_real = np.mean(filtered_real)
            std_dev_real = np.std(filtered_real)
            local_energy_real = np.sum(filtered_real**2)
            features.extend([mean_real, std_dev_real, local_energy_real])
    return features


def apply_gabor_filters_cv(image: np.ndarray, kernels: list) -> list:
    '''
    Apply Gabor filters using OpenCV and compute statistical features.

    Parameters:
        image (np.ndarray): Input image.
        kernels (list): List of Gabor kernels.

    Returns:
        list: List of statistical features for each filter.
    '''
    features = []
    for kernel in kernels:
        filter_img = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        mean_real = np.mean(filter_img)
        std_dev_real = np.std(filter_img)
        local_energy_real = np.sum(filter_img**2)
        features.extend([mean_real, std_dev_real, local_energy_real])
    return features


def gabor_filter_ski(image: np.ndarray, thetas: list[float], frequencies: list[float], avg_width: float = 604.47125, avg_height: float = 961.69775) -> list:
    '''
    Resize and convert image to grayscale, then generate and apply Gabor filters using skimage to the image and compute 
    statistical features.

    Parameters:
        image (np.ndarray): Input image.
        thetas (list[float]): List of orientations for Gabor filters.
        frequencies (list[float]): List of frequencies for Gabor filters.
        avg_width (float): Average width for resizing.
        avg_height (float): Average height for resizing.

    Returns:
        list: List of statistical features for each filter.
    '''
    img = resize_image(image, avg_width, avg_height)
    gray_image = convert_to_grayscale(img)
    feature_vect = apply_gabor_filters_ski(gray_image, thetas, frequencies)
    return feature_vect


def generate_gabor_kernel_ski(thetas: list[float], frequencies: list[float]):
    '''
    Generate Gabor kernels using Skimage

    Returns:
        list: List of Gabor kernels.
    '''
    kernels = []
    for theta in thetas:
        for frequency in frequencies:
            kernel = filters.gabor_kernel(frequency=frequency, theta=theta)
            kernels.append(np.real(kernel))
    return kernels


def generate_gabor_kernel_cv(thetas: list[float], frequencies: list[float]) -> list:
    '''
    Generate Gabor kernels using OpenCV with specified thetas and frequencies.

    Parameters:
        thetas (list[float]): List of orientations for Gabor filters.
        frequencies (list[float]): List of frequencies for Gabor filters.

    Returns:
        list: List of Gabor kernels.
    '''
    kernels = []
    ksize = 70
    for theta in thetas:
        for freq in frequencies:
            kernel = cv2.getGaborKernel(
                (ksize, ksize), 1/freq, theta, 1/freq, 0, ktype=cv2.CV_32F)
            kernels.append(kernel)
    return kernels


def get_gabor_vect(image: np.ndarray, kernels: list, avg_width: float = 604.47125, avg_height: float = 961.69775) -> list:
    '''
    Resize and convert image to grayscale, then apply Gabor filters using OpenCV to the image and compute 
    statistical features.

    Parameters:
        image (np.ndarray): Input image.
        kernels (list): List of Gabor kernels.
        avg_width (float): Average width for resizing.
        avg_height (float): Average height for resizing.

    Returns:
        list: List of statistical features for each filter.
    '''
    img = resize_image(image, avg_width, avg_height)
    gray_image = convert_to_grayscale(img)
    feature_vect = apply_gabor_filters_cv(gray_image, kernels)
    return feature_vect
