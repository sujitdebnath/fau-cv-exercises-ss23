#!/usr/bin/env python3
import sys
sys.path.append(".")
import os
import cv2
import ex0
import ex2
import glob
import fargv  #  pip install fargv or comment line 49

parameters = {
    "images_glob": "./data/ex2/*jpg",
    "ransac_points": 4,
    "features_per_image": 500,
    "max_iterations": 1000,
    "ransac_threshold": 2.0,
    "significant_descriptor_threshold": .7,
    "ransac_points": 4,
    "reference_name": "4.jpg",
    "rendering_order": ["4.jpg", "3.jpg", "5.jpg", "2.jpg", "6.jpg", "1.jpg", "7.jpg", "0.jpg", "8.jpg"],
    "scale": 0.3,
    "output_path": "results/ex2/output.png"
}


def new_main(parameters):
    load_img = lambda x: cv2.resize(cv2.imread(x), None, fx=parameters["scale"], fy=parameters["scale"])
    filenames = glob.glob(parameters["images_glob"])  #  No guaranty of order
    images = {os.path.basename(path): load_img(path) for path in filenames}
    features = {path: ex2.extract_features(images[path], parameters["features_per_image"]) for path in images.keys()}
    points = {path: features[0] for path, features in features.items()}
    descriptors = {path: features[1] for path, features in features.items()}
    names = list(sorted(images.keys()))
    homographies = {}
    for n in range(1, len(names)):
        name1 = names[n-1]
        name2 = names[n]
        homographies[(name1, name2)] = ex2.ransac((points[name1], descriptors[name1]),
                                                  (points[name2], descriptors[name2]),
                                                  steps=parameters["max_iterations"], distance_threshold=parameters["ransac_threshold"],
                                                  similarity_threshold=parameters["significant_descriptor_threshold"],
                                                  n_points=parameters["ransac_points"])
    panorama = ex2.create_stitched_image(images, homographies, parameters["reference_name"], parameters["rendering_order"])
    ex0.show_images([panorama], ["output"])
    ex0.save_images([panorama], [parameters["output_path"]])


if __name__ == "__main__":
    parameters, _ = fargv.fargv(parameters, return_type="dict")  #  Remove
    new_main(parameters)