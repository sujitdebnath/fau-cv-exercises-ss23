import sys
sys.path.append(".")

import re
import pytest
import numpy as np
import cv2
import ex2

epsilon = 1e-6


def all_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon."""
    delta = (t1 - t2) ** 2
    correct = delta > epsilon
    return correct.reshape(-1).mean() == 0


def test_authors():
    author_dict = ex2.__authors__
    assert 0 < len(author_dict) < 3
    fauid_re = re.compile('^[a-z]{2}[0-9]{2}[a-z]{4}$')
    assert all([len(fauid_re.findall(fid)) == 1 for fid in author_dict.keys()])


image = np.zeros([240, 320, 3], dtype=np.uint8)
contours_1 = np.array([[50, 50], [50, 150], [150, 150], [150, 50]])
image1 = cv2.fillPoly(image.copy(), pts =[contours_1], color=(255, 255, 255))
contours_2 = np.array([[100, 100], [300, 100], [50, 200]])
image2 = cv2.fillPoly(image1.copy(), pts =[contours_2], color=(128, 128, 128))


@pytest.mark.parametrize("img_count_bbox", [(image, 0, (0, 0, 0, 0)),
                                            (image1, 9, (50, 50, 150, 150)),
                                            (image2,  57, (50, 50, 300, 200))])
def test_features(img_count_bbox):
    img, count, (left, top, right, bottom) = img_count_bbox
    points, descriptors = ex2.extract_features(img, 500)
    assert count == len(points)
    if len(points):
        assert (points[:, 0] > left).all()
        assert (points[:, 0] < right).all()
        assert (points[:, 1] > top).all()
        assert (points[:, 1] < bottom).all()


@pytest.mark.parametrize("translation", [np.array([0, 0]), np.array([50, 0]), np.array([0, 50]), np.array([50, 50])])
def test_filter_and_align_descriptors(translation):
    image = (np.random.rand(500, 500, 3) * 5).astype(np.uint8)
    contours = np.array([[100, 100], [300, 100], [50, 200]])
    image1 = cv2.fillPoly(image.copy(), pts=[contours], color=(255, 255, 255))
    image2 = cv2.fillPoly(image.copy(), pts=[contours + translation], color=(128, 128, 128))
    points, descriptors = ex2.extract_features(image1, 18)
    translated_points, translated_descriptors = ex2.extract_features(image2, 18)
    points, translated_points = ex2.filter_and_align_descriptors((points, descriptors), (translated_points, translated_descriptors), similarity_threshold=.7)
    assert ((points + translation - translated_points) ** 2 < 60).all()


points1src = np.array(((1, 1), (3, 7), (2, -5), (10, 11)))
points1dst = np.array(((25, 156), (51, -83), (-144, 5), (345, 15)))
points2src = np.array(((1, 1), (1, 1), (3, 7), (2, -5), (10, 11)))
points2dst = np.array(((25, 156), (25, 156), (51, -83), (-144, 5), (345, 15)))
points3src = np.repeat(points2src, 100, axis=0) + np.random.rand(500, 2) * .00001
points3dst = np.repeat(points2dst, 100, axis=0) + np.random.rand(500, 2) * .00001

@pytest.mark.parametrize("src_dst", [(points1src, points1dst), (points2src, points2dst), (points3src, points3dst)])
def test_homography(src_dst):
    points1, points2 = src_dst
    h = ex2.compute_homography(points1, points2)
    h_ref = np.array([[-151.2372466105457, 36.67990057507507, 130.7447340624461],
                      [-27.31264543681857, 10.22762978292494, 118.0943169422209],
                      [-0.04233528054472634, -0.3101691983762523, 1]])
    assert all_similar(h, h_ref)



def test_ransac():
    N = 256
    h = np.array([[-151.2372466105457, 36.67990057507507, 130.7447340624461],
                      [-27.31264543681857, 10.22762978292494, 118.0943169422209],
                      [-0.04233528054472634, -0.3101691983762523, 1]])

    points = np.random.rand(N, 2) * 100
    descriptors = np.random.rand(N, 32)

    moved_points = cv2.perspectiveTransform(points[None, :, :], h)[0, :, :]

    #  Making half the points outliers
    location_noise = np.random.rand(N, 2) * 200 - 10
    outlier = (np.random.rand(N) > .9)[:, None]
    moved_points = moved_points + outlier * location_noise

    new_h = ex2.ransac((points, descriptors), (moved_points, descriptors), steps=10, distance_threshold=.5)
    assert all_similar(h, new_h)


@pytest.mark.parametrize("translation", [np.array([0, 0]), np.array([30, 0]), np.array([0, 20]), np.array([40, 70])])
def test_translate_homography(translation):
    points1 = np.array(((1, 1), (3, 7), (2, -5), (10, 11)))
    points2 = np.array(((25, 156), (51, -83), (-144, 5), (345, 15)))
    homographies = {("src", "dst"): ex2.compute_homography(points1, points2)}
    points = np.random.rand(1000, 2) * 100
    translated_homographies = ex2.translate_homographies(homographies, dx=translation[0], dy=translation[1])
    moved_points = cv2.perspectiveTransform(points[None, :, :], homographies[("src", "dst")])[0, :, :]
    translated_moved_points = cv2.perspectiveTransform(points[None, :, :], translated_homographies[("src", "dst")])[0, :, :]
    assert (((moved_points + translation[None, :]) - translated_moved_points) ** 2 < 1).all()
