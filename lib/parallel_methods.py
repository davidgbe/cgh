import numpy as np
from math import cos

def compute_phase(x, y, z, x_p, y_p, z_p, k):
    return cos(k * np.sqrt((x - x_p)**2 + (y - y_p)**2 + (z - z_p)**2))

def generate_partial_interference_pattern(args):
    (holo_indices, source_indices, holo_image_shape, source_image, k) = args
    partial_holo_image = np.zeros(holo_image_shape[0] * holo_image_shape[1]).reshape(holo_image_shape)
    total = len(holo_indices)
    i = 0
    for (x_p, y_p, z_p, img_i_p, img_j_p) in holo_indices:
        i += 1
        print("%s percent complete" % str(i / total * 100)[:4])
        for (x, y, z, img_i, img_j) in source_indices:
            partial_holo_image[img_i_p, img_j_p] += (compute_phase(x, y, z, x_p, y_p, z_p, k) * source_image[img_i, img_j])
    return partial_holo_image
