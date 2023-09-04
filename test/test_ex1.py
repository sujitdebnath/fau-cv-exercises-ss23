import pytest
import numpy as np
import sys
import cv2
import re
try:
    from line_profiler import LineProfiler  # pip install line_profiler
except Exception as e:
    class LineProfiler():
        def runcall(self, *args):
            raise Exception("LineProfiler not installed")
        def add_function(self, *args):
            raise Exception("LineProfiler not installed")

sys.path.append(".")
import ex1


epsilon = .0001


def all_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon."""
    delta = (t1 - t2) ** 2
    correct = delta > epsilon
    return correct.reshape(-1).mean() == 0


def test_authors():
    author_dict = ex1.__authors__
    assert 0 < len(author_dict) < 3
    fauid_re = re.compile('^[a-z]{2}[0-9]{2}[a-z]{4}$')
    assert all([len(fauid_re.findall(fid))==1 for fid in author_dict.keys()])


image = np.zeros([240, 320, 3], dtype=np.uint8)
contours_1 = np.array([[50,50], [50, 150], [150, 150], [150, 50]])
image1 = cv2.fillPoly(image.copy(), pts =[contours_1], color=(255, 255, 255))
contours_2 = np.array([[100,100], [300, 100], [50, 200]])
image2 = cv2.fillPoly(image1.copy(), pts =[contours_2], color=(128, 128, 128))

float_image = image2[:, :, 0].astype(np.float32)/255.

@pytest.mark.parametrize("img_mean_dev", [(image, 0, 0), (image1, -0.0898592, 0.85003), (image2,  -0.0649, 0.7036)])
def test_harris(img_mean_dev):
    img, R_mean, R_std = img_mean_dev[0][:, :, 0].astype(np.float32)/255., img_mean_dev[1], img_mean_dev[2]
    R, _, _, _, _, _ = ex1.compute_harris_response(img, 0.06)
    assert (R.mean() - R_mean) ** 2 < epsilon
    assert (R.std() - R_std) ** 2 < epsilon


def test_harris_vectorization():
    lp = LineProfiler()
    lp.add_function(ex1.compute_harris_response)
    lp.runcall(ex1.compute_harris_response, float_image, 0.06)
    stats = lp.get_stats()
    assert all([count == 1 for _, count, _ in list(stats.timings.values())[0]])


@pytest.mark.parametrize("img_pointcount", [(image, 0), (image1, 4), (image2,  17)])
def test_points(img_pointcount):
    img, pointcount = img_pointcount[0][:, :, 0].astype(np.float32)/255., img_pointcount[1]
    R, _, _, _, _, _ = ex1.compute_harris_response(img, 0.06)
    points = ex1.detect_corners(R)
    assert len(points[0]) == len(points[1]) == pointcount


def test_point_vectorization():
    lp = LineProfiler()
    lp.add_function(ex1.detect_corners)
    lp.runcall(ex1.detect_corners, float_image, 0.1)
    stats = lp.get_stats()
    assert all([count <= 16 for _, count, _ in list(stats.timings.values())[0]])


# Depending on the 'lesser' or 'less than equal' in the function 'detect_edges', results can be really different.
# This test case allows both the results.
@pytest.mark.parametrize("img_masklength", [(image, (0,0)), (image1, (24, 1576)), (image2,  (431, 2323))])
def test_edges(img_masklength):
    img, edge_sum = img_masklength[0][:, :, 0].astype(np.float32)/255., img_masklength[1]
    R, _, _, _, _, _ = ex1.compute_harris_response(img, 0.06)
    edges = ex1.detect_edges(R)
    assert edges.sum() in edge_sum


def test_edges_vectorization():
    lp = LineProfiler()
    lp.add_function(ex1.detect_edges)
    lp.runcall(ex1.detect_edges, float_image, 0.1)
    stats = lp.get_stats()
    assert all([count < 10 for _, count, _ in list(stats.timings.values())[0]])