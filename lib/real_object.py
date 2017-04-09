import numpy as np
from math import ceil, pi
import multiprocessing as mp
from lib import utilities
from lib.parallel_methods import generate_partial_interference_pattern

class RealObject:
    def __init__(self, image, position_vec, width, height):
        self.image_shape = image.shape
        self.shape = (width, height)
        self.image = image
        self.position_vec = position_vec
        self.scale = (width / self.image_shape[0], height / self.image_shape[1])

    def color_for_ray(self, start, trajec):
        t = (self.position_vec[0] - start[0]) / trajec[0]
        y = start[1] + trajec[1] * t
        z = start[2] + trajec[2] * t
        return self.color_at(y, z)

    def color_given_real_coords(self, y, z):
        y_img = int((y - self.position_vec[1]) / self.scale[0])
        z_img = int((z - self.position_vec[2]) / self.scale[1])
        return self.image[y_img, z_img]

    def color_given_img_coords(self, i, j):
        return self.image[i, j]

    def set_color(self, i, j, color):
        self.image[i, j] = color

    def inc_color(self, i, j, color_inc):
        self.image[i, j] += color_inc

    def iterate_over_points(self, sampling_size=None):
        if sampling_size is None:
            sampling_size = self.shape[0] / self.image_shape[0]
        (width, height) = self.shape
        num_samples_y = ceil(width / sampling_size)
        num_samples_z = ceil(height / sampling_size)

        points = []

        (x, y_start, z_start) = self.position_vec
        for i in range(num_samples_y):
            y = y_start + sampling_size * i
            img_i = int(self.image_shape[0] * i / num_samples_y)
            for j in range(num_samples_z):
                z = z_start + sampling_size * j
                img_j = int(self.image_shape[1] * j / num_samples_z)
                points.append((x, y, z, img_i, img_j))
        return points

    def generate_interference_pattern(self, position_vec, width, height, wavelength=2.5787 * 10**-5, sampling_size=.01, holo_scale=.035):
        image_shape = (ceil(width / holo_scale), ceil(height / holo_scale))
        image = np.zeros(image_shape[0] * image_shape[1]).reshape(image_shape)

        holo_plate = RealObject(image, position_vec, width, height)

        k = 2 * pi / wavelength

        holo_plate_indices = holo_plate.iterate_over_points()
        source_points = self.iterate_over_points(sampling_size)

        core_num = utilities.get_num_cores()
        chunk_size = ceil(float(len(holo_plate_indices)) / core_num)
        
        partial_inteference_args = [(holo_plate_indices[i:i+chunk_size], source_points, image_shape, self.image, k) for i in range(0, len(holo_plate_indices), chunk_size)]
        
        pool = utilities.create_pool()
        partial_holo_plates = pool.map_async(generate_partial_interference_pattern, partial_inteference_args).get()
        pool.close()
        pool.join()

        for i in range(0, len(partial_holo_plates)):
            image = np.add(image, np.mat(partial_holo_plates[i]))

        image = image / image.max() * 255
        holo_plate.image = image
        return holo_plate








