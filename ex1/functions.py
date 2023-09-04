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


    # Step 2: Ixx Iyy Ixy from Idx and Idy (3 Lines)


    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur (5 Lines)


    #Step 4:  Compute the harris response with the determinant and the trace of T (see announcement) (4 lines)


    raise NotImplementedError


def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1 (recommended) : pad the response image to facilitate vectorization (1 line)


    # Step 2 (recommended) : create one image for every offset in the 3x3 neighborhood (6 lines).


    # Step 3 (recommended) : compute the greatest neighbor of every pixel (1 line)


    # Step 4 (recommended) : Compute a boolean image with only all key-points set to True (1 line)


    # Step 5 (recommended) : Use np.nonzero to compute the locations of the key-points from the boolean image (1 line)


    raise NotImplementedError


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
