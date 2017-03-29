import os
import struct
import numpy as np
import time
import sys
from lib import utilities

class MNIST:
    @staticmethod
    def get_image():
        return MNIST.load_images_dataset('../datasets/train-images-idx3-ubyte')

    @staticmethod
    def load_images_dataset(rel_path, limit=1):
        print('Loading image dataset...')
        start = time.time()

        images_file = open(utilities.file_path(__file__, rel_path), 'rb')
        (mag, num_examples, rows, cols) = MNIST.read(images_file, 16, 'i', 4)
        num_examples = limit if (limit is not None and limit < num_examples) else num_examples

        print('Number of examples: %d' % num_examples)
        print('Rows of pixels per image: %d' % rows)
        print('Columns of pixels per image: %d' % cols)

        raw_images = MNIST.read_bytes(images_file, num_examples * rows * cols)
        vec_func = np.vectorize(MNIST.convert_to_unsigned_int)
        raw_images = np.mat([ vec_func(np.array(raw_images[i:i + rows * cols])) for i in range(0, len(raw_images), rows * cols) ])
        images_file.close()

        end = time.time()
        print('Images loaded in %d s' % (end - start))
        return raw_images.reshape(rows, cols)

    @staticmethod
    def read_ints(file, size):
        return MNIST.read(file, size, 'i', 4)

    @staticmethod
    def read_bytes(file, size):
        return MNIST.read(file, size, 'c', 1)

    @staticmethod
    def read(file, size, format, format_byte_size):
        bytes_read = bytes(file.read(size))
        output_size = int(size / format_byte_size)
        return struct.unpack('>'  + format * output_size, bytes_read)

    @staticmethod
    def convert_to_unsigned_int(char):
        return 0 if char == b'' else ord(char)
    
