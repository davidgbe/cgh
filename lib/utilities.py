import numpy as np
import os
import multiprocessing as mp
#import matplotlib.pyplot as plt

def to_grayscale(recon, original):
    return np.multiply(recon, original.std(0)) + original.mean(0)

def file_path(curr_file, *path_elements):
    dir = os.path.dirname(curr_file)
    return os.path.join(dir, *path_elements)

# def save_plot(name):
#     plt.savefig(file_path(__file__, '../images/%s.png' % name))

# def save_image(image_data, name):
#     plt.imshow(image_data, interpolation='nearest', cmap='gray')
#     save_plot(name)

# def save_scatter(name, Y, X=None):
#     if X is None:
#         X = [i for i in range(len(Y))]
#     plt.plot(X, Y, 'ro')
#     save_plot(name)
#     plt.clf()

def bucket(data, bucket_size):
    return [ np.mean(data[i:i+bucket_size]) for i in range(0, len(data), bucket_size) ]

def get_num_cores():
    return mp.cpu_count() - 1

def create_pool(cores=None):
    cores = get_num_cores() if cores is None else cores
    # cores = cores - 10 if cores >= 16 else cores
    print('creating %i threads' % cores)
    return mp.get_context('fork').Pool(cores)

# http://pythonforbiologists.com/index.php/measuring-memory-usage-in-python/
def get_memory():
    vals = psutil.virtual_memory()
    percent_use = vals.available / vals.total
    return percent_use

def print_memory():
    print('Percent memory usage:')
    print(1.0 - get_memory())
