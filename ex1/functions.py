import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

    # Step 1: Compute dx and dy with cv2.Sobel. (2 Lines)
    Idx = cv2.Sobel(I, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    Idy = cv2.Sobel(I, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

    # Step 2: Ixx Iyy Ixy from Idx and Idy (3 Lines)
    Ixx = np.square(Idx)
    Iyy = np.square(Idy)
    Ixy = Idx * Idy

    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur (5 Lines)
    sigma = 1.0
    k_size = 3
    A = cv2.GaussianBlur(Ixx, (k_size, k_size), sigma)
    B = cv2.GaussianBlur(Iyy, (k_size, k_size), sigma)
    C = cv2.GaussianBlur(Ixy, (k_size, k_size), sigma)

    # Step 4:  Compute the harris response with the determinant and the trace of T (see announcement) (4 lines)
    det_T = A * B - C**2
    trace_T = A + B
    R = det_T - k * trace_T**2

    return R, A, B, C, Idx, Idy


def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1: pad the response image to facilitate vectorization (1 line)
    Rnorm = R / R.max()
    R_padded = np.pad(Rnorm, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Step 2: create one image for every offset in the 3x3 neighborhood (6 lines).
    R_offsets_shape = (R.shape[0], R.shape[1], 3, 3)
    R_offsets = np.lib.stride_tricks.as_strided(
        R_padded,
        shape = R_offsets_shape,
        strides = (R_padded.strides[0], R_padded.strides[1], R_padded.strides[0], R_padded.strides[1]),
        writeable = False
    )

    # Step 3: compute the greatest neighbor of every pixel (1 line)
    max_neighbors = np.max(R_offsets, axis=(2,3))

    # Step 4: Compute a boolean image with only all key-points set to True (1 line)
    is_keypoint = (Rnorm.T==max_neighbors.T) & (Rnorm.T>threshold).astype(np.int16)

    # Step 5: Use np.nonzero to compute the locations of the key-points from the boolean image (1 line)
    keypoints = np.nonzero(is_keypoint)

    return keypoints


def detect_edges(R: np.array, edge_threshold: float = -0.01, epsilon=-.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """
    # Step 1 (recommended) : pad the response image to facilitate vectorization (1 line)


    # Step 2 (recommended) : Calculate significant response pixels (1 line)


    # Step 3 (recommended) : create two images with the smaller x-axis and y-axis neighbors respectively (2 lines).


    # Step 4 (recommended) : Calculate pixels that are lower than either their x-axis or y-axis neighbors (1 line)


    # Step 5 (recommended) : Calculate valid edge pixels by combining significant and axis_minimal pixels (1 line)


    raise NotImplementedError
