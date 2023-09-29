import numpy as np
import open3d as o3d
from PIL import Image
import cv2
from typing import List, Tuple, Union

t_points = np.array
t_image = np.array
t_color = Union[List[int], Tuple[int, int, int]]

t_image_list = List[t_image]
t_str_list = List[str]
t_image_triplet = Tuple[t_image, t_image, t_image]

t_image = np.array
t_pointcloud = Union[Tuple[np.array],Tuple[np.array, np.array]]  # only point or points and colors
t_lineset = Union[Tuple[np.array, np.array, np.array], Tuple[np.array, np.array]]  # points, connections, colors
t_3d_drawable = Union[np.array, t_pointcloud, t_lineset]

def o3d_from_drawable(drawable:t_3d_drawable)->Union[o3d.geometry.PointCloud, o3d.geometry.LineSet]:
    if isinstance(drawable, np.ndarray) and len(drawable.shape) == 2 and drawable.shape[1] == 3: # point cloud with no colors
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(drawable)
        point_cloud.colors = o3d.utility.Vector3dVector(np.zeros_like(drawable)+.5)  # gray is the default color
        return point_cloud
    elif isinstance(drawable, tuple) and len(drawable) == 2 and all([isinstance(d, np.ndarray) for d in drawable]) and drawable[0].shape == drawable[1].shape and len(drawable[0].shape)==2 and  drawable[0].shape[1] == 3:  # pointcloud with colors
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(drawable[0])
        point_cloud.colors = o3d.utility.Vector3dVector(drawable[1])
        return point_cloud
    elif isinstance(drawable, tuple) and len(drawable) == 2 and all([isinstance(d, np.ndarray) for d in drawable]) and len(drawable[0].shape) == len(drawable[0].shape) == 2 and drawable[0].shape[1] == drawable[1].shape[1] + 1 == 3:  # lineset with no colors
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(drawable[0])
        line_set.lines = o3d.utility.Vector2iVector(drawable[1])
        line_set.colors = o3d.utility.Vector3dVector(np.zeros([drawable[1].shape[0], 3]) + .5)  # default color is gray
        return line_set
    elif isinstance(drawable, tuple) and len(drawable) == 3 and all([isinstance(d, np.ndarray) for d in drawable]) and all([len(d.shape)==2 for d in drawable]) and [d.shape[1] for d in drawable] == [3, 2, 3]:  # lineset with colors
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(drawable[0])
        line_set.lines = o3d.utility.Vector2iVector(drawable[1])
        line_set.colors = o3d.utility.Vector3dVector(drawable[2])
        return line_set
    else:
        raise ValueError()


def img_to_pointcloud(img: t_image, depth=1.0) -> t_pointcloud:
    _X = np.linspace(0, img.shape[1], img.shape[1])
    _Y = np.linspace(0, img.shape[0], img.shape[0])
    X, Y = np.meshgrid(_X, _Y)
    points = np.concatenate([X.reshape(-1)[:, None], Y.reshape(-1)[:, None], np.zeros([X.size, 1]) + depth], axis=1)
    colors = img.reshape([-1, 3])/255.
    return points, colors


def create_camera_lineset(pointcloud_locations: np.array, camera_location_list, color_list):

    cameras = np.array(camera_location_list)
    colors = np.array(color_list)
    assert len(cameras.shape) == len(colors.shape) == 2
    assert cameras.shape[1] == colors.shape[1] == 3
    assert cameras.shape[0] == colors.shape[0]
    pointcloud_sz = pointcloud_locations.shape[0]

    joined_camera_and_points = np.empty([pointcloud_locations.shape[0]+len(camera_location_list), 3])
    joined_camera_and_points[:len(camera_location_list),:] = camera_location_list
    pointcloud_idx = np.arange(pointcloud_locations.shape[0]) + len(camera_location_list)

    lines = np.zeros([pointcloud_locations.shape[0]*len(camera_location_list), 2], dtype=np.int32)
    line_colors = np.zeros([pointcloud_locations.shape[0] * len(camera_location_list), 3])

    for n, (camera, color) in enumerate(zip(camera_location_list, color_list)):
        joined_camera_and_points[n, :] = cameras[n, :]
        lines[n*pointcloud_sz:(n+1)*pointcloud_sz, 0] = pointcloud_idx
        lines[n * pointcloud_sz:(n + 1) * pointcloud_sz, 1] = n
        line_colors[n * pointcloud_sz:(n + 1) * pointcloud_sz, :] = colors[n:n+1, :]
    joined_camera_and_points[len(camera_location_list):, :] = pointcloud_locations
    return joined_camera_and_points, lines, line_colors


def show_pointclouds(pointcpoulds: List[t_pointcloud]):
    pointclouds = []
    for datum in pointcpoulds:
        if isinstance(datum, np.ndarray) and datum.shape[1] != 3:  # point cloud was an plain image
            points, colors = img_to_pointcloud(datum)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            pointclouds.append(point_cloud)
        elif isinstance(datum, tuple) and len(datum) == 2 and datum[0].shape[1] == datum[1].shape[1] == 3:
            points, colors = datum
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            pointclouds.append(point_cloud)
        elif isinstance(datum, np.ndarray) and len(datum.shape) == 2 and datum.shape[1] == 3:
            points = datum
            colors = np.zeros_like(points)+.5
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            pointclouds.append(point_cloud)
        else:
            raise ValueError
    o3d.visualization.draw_geometries(pointclouds)


def render_correspondence(image1: t_image, points1: t_points, image2:t_image, points2:t_points, color:t_color=[0, 255, 0],
                          scale: float=1., h_align: bool=True, thickness: int=1) -> t_image:
    assert points1.shape == points2.shape
    if not hasattr(scale, "__len__"):
        scale = (scale, scale)
    # image1 = cv2.resize(image1, None, fx=scale[0], fy=scale[1])
    # image2 = cv2.resize(image2, None, fx=scale[0], fy=scale[1])
    # points1 *= np.array(scale)[None, :]
    # points2 *= np.array(scale)[None, :]
    points1, points2 = points1.astype(np.int), points2.astype(np.int)

    if h_align:
        correspondence = np.concatenate([image1, image2], axis=1)
        points2[:, 0] += image1.shape[1]
    else:
        correspondence = np.concatenate([image1, image2], axis=0)
        points2[:, 1] += image1.shape[0]

    for n in range(points1.shape[0]):
        cv2.line(correspondence, tuple(points1[n, :]), tuple(points2[n, :]), color, thickness)
    correspondence = cv2.resize(correspondence, None, fx=scale[0], fy=scale[1])
    return correspondence


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
