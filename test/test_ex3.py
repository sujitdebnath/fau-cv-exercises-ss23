import sys
sys.path.append(".")
import pdb

import re
import pytest
import numpy as np
import cv2
import ex2
import ex3

epsilon = 1e-6

intrinsic_camera_params = np.array([[2890., 0, 1440],
                                        [0, 2890, 960],
                                        [0, 0, 1]])


def all_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon."""
    delta = (t1 - t2) ** 2
    correct = delta > epsilon
    return correct.reshape(-1).mean() == 0


def is_translated(t1, t2, t_ref):
    if np.linalg.norm(t1 - t_ref) < epsilon:
        if np.linalg.norm(t2 + t_ref) < epsilon:
            assert True
        else:
            assert False
    elif np.linalg.norm(t1 + t_ref) < epsilon:
        if np.linalg.norm(t2 - t_ref) < epsilon:
            assert True
        else:
            assert False


def is_rotated(R1, R2, R1_ref, R2_ref):
    if np.linalg.norm(R1 - R1_ref) < epsilon:
        if np.linalg.norm(R2 - R2_ref) < epsilon:
            assert True
        else:
            assert False
    elif np.linalg.norm(R1 - R2_ref) < epsilon:
        if np.linalg.norm(R2 - R1_ref) < epsilon:
            assert True
        else:
            assert False


def test_authors():
    author_dict = ex3.__authors__
    assert 0 < len(author_dict) < 3
    fauid_re = re.compile('^[a-z]{2}[0-9]{2}[a-z]{4}$')
    assert all([len(fauid_re.findall(fid)) == 1 for fid in author_dict.keys()])

def test_fundamental_matrix():
    points1 = np.array([(1, 1), (3, 7), (2, -5), (10, 11), (11, 2), (-3, 14), (236, -514), (-5, 1)])
    points2 = np.array([(25, 156), (51, -83), (-144, 5), (345, 15),
                                    (215, -156), (151, 83), (1544, 15), (451, -55)])

    F = ex3.compute_fundamental_matrix(points1, points2)
    F_ref = np.array([[0.001260822171230067,  0.0001614643951166923, -0.001447955678643285],
                 [-0.002080014358205309, -0.002981504896782918, 0.004626528742122177],
                 [-0.8665185546662642,   -0.1168790312603214,   1]])
    assert all_similar(F, F_ref)

def test_triangulate():

    P1 = np.array([[1, 2, 3, 6], [4, 5, 6, 37], [7, 8, 9, 15]]).astype('float')
    P2 = np.array([[51, 12, 53, 73], [74, 15, -6, -166], [714, -8, 95, 16]]).astype('float')

    wp = ex3.triangulate(P1, P2, (14.0, 267.0), (626.0, 67.0))
    wpref = [0.782409, 3.89115, -5.72358]

    assert all_similar(wp, wpref)

def test_essential_matrix():
    F_ref = np.array([[0.001260822171230067, 0.0001614643951166923, -0.001447955678643285],
                      [-0.002080014358205309, -0.002981504896782918, 0.004626528742122177],
                      [-0.8665185546662642, -0.1168790312603214, 1]])

    E = ex3.compute_essential_matrix(F_ref, intrinsic_camera_params)

    E_ref = np.array([[-2.54228083, -0.32557156, -1.37388009],
           [4.19407331, 6.01180952, 4.08355499],
           [0.73101932, 1.91632705, 1.]])

    assert all_similar(E, E_ref)


def test_decompose():
    E = np.array([[3.193843825323984, 1.701122615578195, -6.27143074201245],
                  [-41.95294882825382, 127.76771801644763, 141.5226870527082],
                  [-8.074440557013252, -134.9928434422784, 1]])
    R1, R2, t1, t2 = ex3.decompose(E)

    t_ref = np.array([-0.9973654079783344, -0.04579551552933408, -0.05625845470338976])
    R1_ref = np.array([[0.9474295893819155, -0.1193419720472381, 0.2968748336782551],
                       [0.2288481582901039, 0.9012012887804273, -0.3680553729369057],
                       [-0.2236195286884464, 0.4166458097813701, 0.8811359574894123]])
    R2_ref = np.array([[0.9332689072231527, 0.01099472166665292, 0.3590101153254257],
                       [-0.1424930907107563, -0.9061762138657301, 0.3981713055001164],
                       [0.329704209724715, -0.4227573601008549, -0.8441394129942975]])

    is_translated(t1, t2, t_ref)
    is_rotated(R1, R2, R1_ref, R2_ref)
