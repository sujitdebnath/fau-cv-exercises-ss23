import numpy as np
import os
import cv2
import sys
sys.path.append(".")
import ex1


if __name__ == "__main__":
    input_path = os.path.join("data", "ex1", "input.jpg")
    input = cv2.imread(input_path, cv2.IMREAD_COLOR)
    gray_float_image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    R, A, B, C, Idx, Idy = ex1.compute_harris_response(gray_float_image, k=.06)
    image_names = ["R", "A", "B", "C", "Idx", "Idy"]
    # 0 Centered images are shifted from black to gray for 0. by adding .5
    ex1.show_images([input, .5 + R, A, B, .5 + C, .5 + Idx, .5 + Idy], ["input"] + image_names, tile_yx=(400, 400))
    output_paths = [os.path.join("results", "ex1", f"{name}.png") for name in image_names]
    ex1.save_images([.5 + R, A, B, .5 + C, .5 + Idx, .5 + Idy], output_paths)

    points = ex1.detect_corners(R, threshold=.1)
    drawn_points = ex1.draw_points(input, points, color=(0, 255, 0))
    ex1.show_images([input, .5 + R, drawn_points], ["Input", "Harris Response", "Key points"], tile_yx=(400, 400))
    ex1.save_images([drawn_points], [os.path.join("results", "ex1", "points.png")])

    edges = ex1.detect_edges(R, edge_threshold=-.01)
    drawn_edges = ex1.draw_mask(input, edges, color=(255, 0, 0))
    ex1.show_images([.5 + R, edges, drawn_edges], ["Harris Response", "Edges", "Drawn Edges"], tile_yx=(400, 400))
    ex1.save_images([edges, drawn_edges], [os.path.join("results", "ex1", f) for f in ["edges.png", "drawn_edges.png"]])

