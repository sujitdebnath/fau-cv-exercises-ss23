#!/usr/bin/env python3

from PIL import Image
import fargv
import sys
sys.path.append(".")
import numpy as np
from matplotlib import pyplot as plt
import ex4

p = {
        "left": "./data/ex4/tsukuba_1.png",
        "right": "./data/ex4/tsukuba_2.png",
        "sigma_x": 1.,
        "sigma_z": 1.,
        "median_filter_size": 3,
     }


if __name__ == "__main__":
    # loading the parameters
    p, _ = fargv.fargv(p)

    # loading the images
    left_img = np.asarray(Image.open(p.left))
    right_img = np.asarray(Image.open(p.right))

    # computing the maximum translation between the two images OR
    # amount of pixels the camera translated while taking the second picture.
    max_translation = ex4.get_max_translation(left_img, right_img)
    print("Max translation:", max_translation)

    dm_filtered = ex4.disparity_map(left_img,
                                 right_img,
                                 pad_size=np.abs(max_translation),
                                 offset=np.abs(max_translation),
                                 sigma_x=p.sigma_x,
                                 sigma_z=p.sigma_z, median_filter_size=p.median_filter_size)

    # Visualizing the disparity map
    plt.figure()
    plt.subplot(131)
    plt.imshow(left_img, cmap="gray")
    plt.subplot(132)
    plt.imshow(right_img, cmap="gray")
    plt.subplot(133)
    plt.imshow(dm_filtered, cmap="gray")
    plt.axis('off')
    plt.show()

    # Visualizing the translation check
    plt.figure()
    plt.subplot(131)
    plt.imshow(left_img, cmap="gray")
    plt.subplot(132)
    plt.imshow(right_img, cmap="gray")
    plt.subplot(133)

    # Plot disparity uses bilinear_grid_sample (application of sampling field) internally
    d = ex4.plot_disparity(left_img, dm_filtered[:, :left_img.shape[1]])
    plt.imshow(np.uint8(d), cmap="gray")
    plt.show()


