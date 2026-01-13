"""
HOG (Histogram of Oriented Gradients) feature extraction module.
Uses skimage.feature.hog for extracting HOG features.
"""

import numpy as np
import cv2
from skimage.feature import hog
from skimage import color
from typing import List, Optional
from tqdm import tqdm

from config import (
    WINDOW_SIZE,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    HOG_BLOCK_NORM
)


def extract_hog_features(
    image: np.ndarray,
    orientations: int = HOG_ORIENTATIONS,
    pixels_per_cell: tuple = HOG_PIXELS_PER_CELL,
    cells_per_block: tuple = HOG_CELLS_PER_BLOCK,
    block_norm: str = HOG_BLOCK_NORM,
    visualize: bool = False
) -> np.ndarray:
    """
    Extract HOG features from a single image patch.
    
    Args:
        image: Input image (can be color or grayscale)
        orientations: Number of orientation bins
        pixels_per_cell: Size of a cell in pixels
        cells_per_block: Number of cells in each block
        block_norm: Block normalization method
        visualize: If True, return visualization as well
        
    Returns:
        HOG feature vector (and optionally the visualization image)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            # OpenCV uses BGR, convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image
    
    # Ensure the image is the correct size
    if gray.shape != WINDOW_SIZE:
        gray = cv2.resize(gray, (WINDOW_SIZE[1], WINDOW_SIZE[0]))
    
    # Extract HOG features
    if visualize:
        features, hog_image = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            visualize=True,
            feature_vector=True
        )
        return features, hog_image
    else:
        features = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            visualize=False,
            feature_vector=True
        )
        return features


def extract_hog_features_batch(
    images: List[np.ndarray],
    show_progress: bool = True
) -> np.ndarray:
    """
    Extract HOG features from a batch of images.
    
    Args:
        images: List of image patches
        show_progress: Whether to show progress bar
        
    Returns:
        2D array of HOG features (num_images, feature_dim)
    """
    features_list = []
    
    iterator = tqdm(images, desc="Extracting HOG features") if show_progress else images
    
    for image in iterator:
        features = extract_hog_features(image)
        features_list.append(features)
    
    return np.array(features_list)


def extract_color_histogram(
    image: np.ndarray,
    bins: int = 32,
    normalize: bool = True,
    use_hsv: bool = True
) -> np.ndarray:
    """
    Extract color histogram features from an image.
    
    Args:
        image: Input BGR image
        bins: Number of bins per channel
        normalize: Whether to normalize the histogram
        use_hsv: Whether to convert to HSV (better for color discrimination)
        
    Returns:
        Concatenated color histogram feature vector
    """
    if len(image.shape) != 3:
        # Grayscale image, compute single histogram
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        if normalize:
            hist = hist / (hist.sum() + 1e-7)
        return hist.flatten()
    
    histograms = []
    
    # BGR histogram
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        if normalize:
            hist = hist / (hist.sum() + 1e-7)
        histograms.append(hist.flatten())
    
    # HSV histogram (much better for distinguishing characters by color)
    if use_hsv:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Hue is most important for color (0-180 in OpenCV)
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        if normalize:
            hist_h = hist_h / (hist_h.sum() + 1e-7)
            hist_s = hist_s / (hist_s.sum() + 1e-7)
            hist_v = hist_v / (hist_v.sum() + 1e-7)
        histograms.extend([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    
    return np.concatenate(histograms)
    
    # Color image, compute histogram for each channel
    histograms = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        if normalize:
            hist = hist / (hist.sum() + 1e-7)
        histograms.append(hist.flatten())
    
    return np.concatenate(histograms)


def extract_combined_features(
    image: np.ndarray,
    use_color: bool = True,
    color_bins: int = 16
) -> np.ndarray:
    """
    Extract combined HOG and color histogram features.
    
    Args:
        image: Input image
        use_color: Whether to include color histogram
        color_bins: Number of bins for color histogram
        
    Returns:
        Combined feature vector
    """
    # Extract HOG features
    hog_features = extract_hog_features(image)
    
    if use_color and len(image.shape) == 3:
        # Extract color histogram
        color_features = extract_color_histogram(image, bins=color_bins)
        # Combine features
        return np.concatenate([hog_features, color_features])
    
    return hog_features


def extract_combined_features_batch(
    images: List[np.ndarray],
    use_color: bool = True,
    show_progress: bool = True
) -> np.ndarray:
    """
    Extract combined features from a batch of images.
    
    Args:
        images: List of image patches
        use_color: Whether to include color histogram
        show_progress: Whether to show progress bar
        
    Returns:
        2D array of features
    """
    features_list = []
    
    iterator = tqdm(images, desc="Extracting features") if show_progress else images
    
    for image in iterator:
        features = extract_combined_features(image, use_color=use_color)
        features_list.append(features)
    
    return np.array(features_list)


def get_hog_feature_dimension() -> int:
    """
    Calculate the dimension of HOG features for the current configuration.
    
    Returns:
        Dimension of HOG feature vector
    """
    # Create a dummy image and extract features to get the dimension
    dummy_image = np.zeros(WINDOW_SIZE, dtype=np.uint8)
    features = extract_hog_features(dummy_image)
    return len(features)


if __name__ == "__main__":
    # Test HOG feature extraction
    print(f"Window size: {WINDOW_SIZE}")
    print(f"HOG orientations: {HOG_ORIENTATIONS}")
    print(f"Pixels per cell: {HOG_PIXELS_PER_CELL}")
    print(f"Cells per block: {HOG_CELLS_PER_BLOCK}")
    
    feature_dim = get_hog_feature_dimension()
    print(f"HOG feature dimension: {feature_dim}")
    
    # Test on a random image
    test_image = np.random.randint(0, 255, (*WINDOW_SIZE, 3), dtype=np.uint8)
    
    hog_features = extract_hog_features(test_image)
    print(f"HOG features shape: {hog_features.shape}")
    
    combined_features = extract_combined_features(test_image, use_color=True)
    print(f"Combined features shape: {combined_features.shape}")
