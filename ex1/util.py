import numpy as np
from typing import List, Tuple
import cv2


t_image_list = List[np.array]
t_str_list = List[str]
t_image_triplet = Tuple[np.array, np.array, np.array]


def show_images(images: t_image_list, names: t_str_list, location_yx: Tuple[int, int] = (50, 50),
                tile_yx: Tuple[int, int] = (0, 0), tiles_per_row: int = 4) -> None:
    """Shows one or more images at once.

    Displaying a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        names: A list of strings that will appear as the window titles for each image
        location_yx: A tuple of integers signifying the desired location for the image windows.
        tile_yx: A tuple with the offset of each tile height nd width-wise if set to (0, 0) all images will be rendered
            without window sizes controlled.
        tiles_per_row: An integer defining when to wrap

    Returns:
        None
    """
    assert len(images) == len(names)
    for n in range(len(images)):
        cv2.namedWindow(names[n])
        if images[n].dtype == np.bool:
            cv2.imshow(names[n], images[n].astype(np.float32))
        else:
            cv2.imshow(names[n], images[n])
        x = location_yx[1] + (n % tiles_per_row) * tile_yx[1]
        y = location_yx[0] + (n // tiles_per_row) * tile_yx[0]
        cv2.moveWindow(names[n], x, y)
    cv2.waitKey()
    cv2.destroyAllWindows()


def save_images(images: t_image_list, filenames: t_str_list, **kwargs) -> None:
    """Saves one or more images at once.

    Saving a single image can be done by putting it in a list.
    If the paths have directories, they must already exist.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        filenames: A list of strings where each respective file will be created

    Returns:
        None
    """
    assert len(images) == len(filenames)
    for n in range(len(images)):
        if images[n].dtype == np.uint8:
            cv2.imwrite(filenames[n], images[n], **kwargs)
        elif images[n].dtype in [np.float32, np.float64, np.bool]:
            image = np.clip(images[n], 0, 1)
            image = (image * 255.).astype(np.uint8)
            cv2.imwrite(filenames[n], image, **kwargs)
        else:
            raise ValueError


def draw_mask(input_image: np.array, mask: np.array, color: Tuple[int, int, int]) -> np.array:
    """Draws a boolean image on a color image.

    Args:
        input_image: A BGR byte image
        mask: A boolean image
        color: A tuple with red, green, and blue values in the range [0-255]

    Returns:
        A BGR byte image
    """
    assert input_image.dtype == np.uint8 and len(input_image.shape) == 3
    keep_pixels = ~mask[:, :, None]
    img_b, img_g, img_r = input_image[:, :, : 1], input_image[:, :, 1: 2], input_image[:, :, 2:]
    color_r, color_g, color_b = color
    img_r = img_r * keep_pixels + color_r * mask[:, :, None]
    img_g = img_g * keep_pixels + color_g * mask[:, :, None]
    img_b = img_b * keep_pixels + color_b * mask[:, :, None]
    result = np.concatenate([img_b, img_g, img_r], axis=2).astype(np.uint8)
    return result


def draw_points(input_image: np.array, points: Tuple[np.array, np.array], color: Tuple[int, int, int]) -> np.array:
    """Draws a set of points on a color image.

    Args:
        input_image: A BGR byte image
        points: A tuple of np.array with the integer coordinates of the points to be rendered
        color: A tuple with red, green, and blue values in the range [0-255]

    Returns:
        A BGR byte image
    """
    assert input_image.dtype == np.uint8 and len(input_image.shape) == 3
    x_points, y_points = points
    assert x_points.shape == y_points.shape and len(y_points.shape) == 1
    key_points = [cv2.KeyPoint(float(x_points[n]), float(y_points[n]), 1) for n in range(len(x_points))]
    return cv2.drawKeypoints(input_image, key_points, outImage=None, color=color)
