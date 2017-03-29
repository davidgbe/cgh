import numpy as np
from math import ceil, pi, cos

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

        (x, y_start, z_start) = self.position_vec
        for i in range(num_samples_y):
            y = y_start + sampling_size * i
            img_i = int(self.image_shape[0] * i / num_samples_y)
            for j in range(num_samples_z):
                z = z_start + sampling_size * j
                img_j = int(self.image_shape[1] * j / num_samples_z)
                yield (x, y, z, img_i, img_j)

    def generate_interference_pattern(self, position_vec, width, height, wavelength=2.49 * 10**-5, sampling_size=.007, holo_scale=.007):
        image_shape = (ceil(width / holo_scale), ceil(height / holo_scale))
        image = np.zeros(image_shape[0] * image_shape[1]).reshape(image_shape)

        holo_plate = RealObject(image, position_vec, width, height)

        k = 2 * pi / wavelength

        last_i_p = 0
        for x_p, y_p, z_p, img_i_p, img_j_p in holo_plate.iterate_over_points():
            if last_i_p != img_i_p:
                print(img_i_p)
                last_i_p = img_i_p
            for x, y, z, img_i, img_j in self.iterate_over_points(sampling_size):
                r = np.sqrt((x - x_p)**2 + (y - y_p)**2 + (z - z_p)**2)
                inc = self.color_given_img_coords(img_i, img_j) * cos(k * r)
                holo_plate.inc_color(img_i_p, img_j_p, inc)
        print(holo_plate.image)
        holo_plate.image = holo_plate.image - holo_plate.image.min()
        holo_plate.image = 255 / holo_plate.image.max() * holo_plate.image
        return holo_plate



