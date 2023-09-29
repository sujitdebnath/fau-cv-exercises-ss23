import sys

sys.path.append(".")
import ex4

import pytest
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error

epsilon = 1e-6


def all_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon."""
    delta = (t1 - t2) ** 2
    correct = delta > epsilon
    return correct.reshape(-1).mean() == 0


def _compress_labels(img, max_val=256):
    img = np.uint8(img)
    img_labels = np.unique(img)
    img_compressor = np.arange(max_val+1)
    img_compressor[np.int32(img_labels)] = np.arange(len(img_labels))
    return img_compressor[img], len(img_labels)


def _img_to_onehot(img, max_val=256):
    img = img.ravel()
    img_onehot = np.zeros([img.shape[0], max_val])
    img_onehot[np.arange(img.shape[0], dtype=np.int32), img] = 1
    return img_onehot

def get_label_iou(img1, img2, max_val=256):
    img1, img1_max_label = _compress_labels(img1, max_val=max_val)
    img2, img2_max_label = _compress_labels(img2, max_val=max_val)
    img1 = _img_to_onehot(img1, img1_max_label)
    img2 = _img_to_onehot(img2, img2_max_label)
    label_intersections = np.dot(img1.T, img2)
    label_unions = (img1.sum(axis=0)[:, None] + img2.sum(axis=0)[None, :]) - label_intersections  ## double check
    label_ious = label_intersections/label_unions
    return label_ious, label_intersections, label_unions


img_1 = np.asarray(Image.open("./data/ex4/tsukuba_1.png"))
img_2 = np.asarray(Image.open("./data/ex4/tsukuba_2.png"))
img_3 = np.asarray(Image.open("./data/ex4/tsukuba_3.png"))
img_4 = np.asarray(Image.open("./data/ex4/tsukuba_4.png"))
depth = np.asarray(Image.open("./data/ex4/tsukuba_depth.png"))


@pytest.mark.parametrize("left_right_translation", [(img_1, img_2, -15),
                                                    (img_1, img_3, -29),
                                                    (img_1, img_4, -44),
                                                    (img_3, img_4, -18)])
def test_max_translation(left_right_translation):
    left, right, translation = left_right_translation
    assert ex4.get_max_translation(left, right) == translation


@pytest.mark.parametrize("left_right_disparity_pad_sum", [(img_1, img_2, 3, 15, 1146181.98575),
                                                          (img_1, img_2, 4, 15, 1011708.97066),
                                                          (img_1, img_2, 5, 15, 878151.3641338),
                                                          (img_1, img_2, 6, 15, 966458.7369587),
                                                          (img_1, img_2, 7, 15, 1092038.543961),
                                                      ])
def test_disparity_hypothesis(left_right_disparity_pad_sum):
    left, right, disparity, pad, sum = left_right_disparity_pad_sum
    assert all_similar(ex4.render_disparity_hypothesis(left, right, disparity, pad).sum(), sum)


@pytest.mark.parametrize("left_right_depth_translation_threshold", [(img_2, img_3, depth, -18, 0.42),
                                                                    (img_3, img_4, depth, -18, 0.48)])
def test_disparity_as_segmentation(left_right_depth_translation_threshold):
    left, right, depth, translation, threshold = left_right_depth_translation_threshold
    dm_filtered = ex4.disparity_map(left,
                                 right,
                                 pad_size=np.abs(translation),
                                 offset=np.abs(translation),
                                 sigma_x=1.0,
                                 sigma_z=1.0, median_filter_size=3)
    left_crop = 0
    right_crop = translation
    croped_dm_filtered = dm_filtered[:, left_crop:right_crop]  # unpadding the disparity map
    iou, intersections, unions = get_label_iou(croped_dm_filtered, depth)
    recall=intersections.max(axis=0)/(intersections.sum(axis=0))
    precision = intersections.max(axis=1) / (intersections.sum(axis=1))
    fscore = (2 * precision.mean() * recall.mean()) / (precision.mean() + recall.mean())
    assert fscore > threshold


@pytest.mark.parametrize("left_right_translation_ssim_mse", [(img_1, img_2, -15, .55, .006),
                                                                (img_2, img_3, -18, .55, .005),
                                                                (img_3, img_4, -18, .55, .005)])
def test_reconstruction(left_right_translation_ssim_mse):
    left, right, translation, ssim, mse = left_right_translation_ssim_mse
    dm_filtered = ex4.disparity_map(left,
                                 right,
                                 pad_size=np.abs(translation),
                                 offset=np.abs(translation),
                                 sigma_x=1.0,
                                 sigma_z=1.0, median_filter_size=3)
    left_crop = 0
    right_crop = translation
    croped_dm_filtered = dm_filtered[:, left_crop:right_crop]  # unpadding the disparity map
    reconstructed_right = ex4.plot_disparity(left, croped_dm_filtered).astype(float) / 255.
    right = right.astype(float) / 255.
    rgb_ssim = sum([structural_similarity(reconstructed_right[:, :, n], right[:, :, n]) for n in range(2)]) / 3
    rgb_mse = sum([mean_squared_error(reconstructed_right[:, :, n], right[:, :, n]) for n in range(2)]) / 3
    assert rgb_ssim > ssim
    assert rgb_mse < mse

