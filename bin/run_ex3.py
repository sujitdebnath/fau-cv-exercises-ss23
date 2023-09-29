#!/usr/bin/env python3
import fargv
import os
import sys
sys.path.append(".")

import ex3
import cv2
import numpy as np
import open3d as o3d
import fargv
import time

from ex2 import extract_features
from ex3 import triangulate_all_points
from ex3 import relativeTransformation
from ex3 import fast_filter_and_align_descriptors, compute_essential_matrix
from ex3 import compute_F_ransac

parameters = {
        "img1_path": "./data/ex3/img1.jpg",
        "img2_path": "./data/ex3/img2.jpg",
        "camera_params_list": [2890., 0, 1440, 0, 2890, 960, 0, 0, 1],
        "features_per_image": 5000,
        "correspondence_scale": 0.2,
        "ransac_steps": 10000,
        "ransac_points": 8,
        "ransac_distance_threshold": 4.0,
        "output_path": "results/ex3/",
    }


def main(parameters):
    # parsing command line arguments
    p, _ = fargv.fargv(parameters)
    intrinsic_camera_params = np.reshape(p.camera_params_list, [3, 3]) # defining the intrinsic camera parameters

    # loading the images
    img1 = cv2.imread(p.img1_path)
    img2 = cv2.imread(p.img2_path)

    # extracting ORB features from exercise 2.
    points1, descriptors1 = extract_features(img1, num_features=p.features_per_image)
    points2, descriptors2 = extract_features(img2, num_features=p.features_per_image)

    # faster filtering and aligning descriptors based on Flann matching
    points1, points2 = fast_filter_and_align_descriptors((points1, descriptors1), (points2, descriptors2))
    ex3.show_images([ex3.render_correspondence(img1, points1, img2, points2,
                                               scale=p.correspondence_scale,
                                               thickness=3)], ["Correspondences"])
    ex3.save_images([ex3.render_correspondence(img1, points1, img2, points2,
                                               scale=p.correspondence_scale,
                                               thickness=3)],
                    [os.path.join(p.output_path, "Correspondences_before_ransac.png")])

    # calculating the Fundamental matrix (F) and the inliers using RANSAC
    F, inlier_indexes = compute_F_ransac(points1, points2,
                                         steps=p.ransac_steps,
                                         n_points=p.ransac_points,
                                         distance_threshold=p.ransac_distance_threshold)
    print("RANSAC inliers ", inlier_indexes.shape[0])
    inlier_points1, inlier_points2 = points1[inlier_indexes, :], points2[inlier_indexes, :]
    ex3.show_images([ex3.render_correspondence(img1, inlier_points1, img2, inlier_points2,
                                               scale=p.correspondence_scale,
                                               thickness=2)], ["Correspondences"])
    ex3.save_images([ex3.render_correspondence(img1, inlier_points1, img2, inlier_points2,
                                               scale=p.correspondence_scale,
                                               thickness=2)],
                    [os.path.join(p.output_path, "Correspondences_after_ransac.png")])

    # Compute relative transformation / Essential Matrix
    E = compute_essential_matrix(F, intrinsic_camera_params)

    # camera positioning and View matrices in real world
    View1 = np.eye(3, 4)
    View2 = relativeTransformation(E, inlier_points1, inlier_points2, intrinsic_camera_params)
    print("View2:", View2)

    # triangulate inlier matches
    wps = triangulate_all_points(View1, View2, intrinsic_camera_params, inlier_points1,
                                                    inlier_points2)
    print("Wps size:", wps.shape)

    # finding the camera location for View 2
    large_view = np.eye(4)
    large_view[:3, :4] = View2
    model = np.linalg.inv(large_view)
    camera2_location = (np.dot(model[:3, :3], [[0], [0], [0]]) + model[:3, 3:4]).T[0, :].tolist()

    # drawing the point cloud
    cameras = [[0,0,0], camera2_location]
    colors= [[0, 0, 1.], [1, 0, 0.]]
    np_lineset = ex3.create_camera_lineset(wps, cameras, colors)
    o3d.visualization.draw_geometries([ex3.o3d_from_drawable(dr) for dr in [np_lineset, wps]])

if __name__ == '__main__':
    main(parameters)
