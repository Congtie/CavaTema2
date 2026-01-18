import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage import color
from typing import List, Optional
from tqdm import tqdm

from config import (
    WINDOW_SIZE,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    HOG_BLOCK_NORM,
    LBP_RADIUS,
    LBP_N_POINTS,
    LBP_METHOD,
    LBP_HIST_BINS
)


def extract_hog_features(
    image: np.ndarray,
    orientations: int = HOG_ORIENTATIONS,
    pixels_per_cell: tuple = HOG_PIXELS_PER_CELL,
    cells_per_block: tuple = HOG_CELLS_PER_BLOCK,
    block_norm: str = HOG_BLOCK_NORM,
    visualize: bool = False
) -> np.ndarray:
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image
    
    if gray.shape != WINDOW_SIZE:
        gray = cv2.resize(gray, (WINDOW_SIZE[1], WINDOW_SIZE[0]))
    
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
    features_list = []
    
    iterator = tqdm(images, desc="Extracting HOG features") if show_progress else images
    
    for image in iterator:
        features = extract_hog_features(image)
        features_list.append(features)
    
    return np.array(features_list)


def extract_lbp_features(
    image: np.ndarray,
    radius: int = LBP_RADIUS,
    n_points: int = LBP_N_POINTS,
    method: str = LBP_METHOD,
    n_bins: int = LBP_HIST_BINS
) -> np.ndarray:
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    if gray.shape != WINDOW_SIZE:
        gray = cv2.resize(gray, (WINDOW_SIZE[1], WINDOW_SIZE[0]))

    lbp = local_binary_pattern(gray, n_points, radius, method=method)

    if method == 'uniform':
        n_bins = n_points + 2

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return hist.astype(np.float64)


def extract_color_histogram(
    image: np.ndarray,
    bins: int = 32,
    normalize: bool = True,
    use_hsv: bool = True
) -> np.ndarray:
    if len(image.shape) != 3:
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        if normalize:
            hist = hist / (hist.sum() + 1e-7)
        return hist.flatten()
    
    histograms = []
    
    if use_hsv:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        if normalize:
            hist_h = hist_h / (hist_h.sum() + 1e-7)
            hist_s = hist_s / (hist_s.sum() + 1e-7)
            hist_v = hist_v / (hist_v.sum() + 1e-7)
        histograms.extend([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        return np.concatenate(histograms)
    
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        if normalize:
            hist = hist / (hist.sum() + 1e-7)
        histograms.append(hist.flatten())
    
    return np.concatenate(histograms)


def extract_combined_features(
    image: np.ndarray,
    use_color: bool = True,
    use_lbp: bool = True,
    color_bins: int = 16
) -> np.ndarray:
    feature_parts = []

    hog_features = extract_hog_features(image)
    feature_parts.append(hog_features)

    if use_lbp:
        lbp_features = extract_lbp_features(image)
        feature_parts.append(lbp_features)

    if use_color and len(image.shape) == 3:
        color_features = extract_color_histogram(image, bins=color_bins)
        feature_parts.append(color_features)

    return np.concatenate(feature_parts)


def extract_combined_features_batch(
    images: List[np.ndarray],
    use_color: bool = True,
    use_lbp: bool = True,
    show_progress: bool = True
) -> np.ndarray:
    features_list = []

    iterator = tqdm(images, desc="Extracting features") if show_progress else images

    for image in iterator:
        features = extract_combined_features(image, use_color=use_color, use_lbp=use_lbp)
        features_list.append(features)

    return np.array(features_list)


def get_hog_feature_dimension() -> int:
    dummy_image = np.zeros(WINDOW_SIZE, dtype=np.uint8)
    features = extract_hog_features(dummy_image)
    return len(features)


if __name__ == "__main__":
    print(f"Window size: {WINDOW_SIZE}")
    print(f"HOG orientations: {HOG_ORIENTATIONS}")
    print(f"Pixels per cell: {HOG_PIXELS_PER_CELL}")
    print(f"Cells per block: {HOG_CELLS_PER_BLOCK}")
    print(f"LBP radius: {LBP_RADIUS}, points: {LBP_N_POINTS}, method: {LBP_METHOD}")

    feature_dim = get_hog_feature_dimension()
    print(f"HOG feature dimension: {feature_dim}")

    test_image = np.random.randint(0, 255, (*WINDOW_SIZE, 3), dtype=np.uint8)

    hog_features = extract_hog_features(test_image)
    print(f"HOG features shape: {hog_features.shape}")

    lbp_features = extract_lbp_features(test_image)
    print(f"LBP features shape: {lbp_features.shape}")

    combined_features = extract_combined_features(test_image, use_color=True, use_lbp=True)
    print(f"Combined features shape (HOG+LBP+Color): {combined_features.shape}")
