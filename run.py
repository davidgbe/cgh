from lib.mnist import MNIST
from lib import utilities
from lib.real_object import RealObject
import numpy as np
import multiprocessing as mp
import pickle
import sys
from random import randint as rand

args = sys.argv
write_path = args[1]
output_name = args[2]
depth = float(args[3])
displacement = float(args[4])
sampling_frequency = float(args[5])
holo_resolution = float(args[6])


mp.freeze_support()
image = MNIST.get_image()

ro = RealObject(image, np.zeros(3), 1, 1)
print(sys.argv)
holo = ro.generate_interference_pattern(np.array([depth, displacement, 0]), 1, 1, sampling_size=sampling_frequency, holo_scale=holo_resolution)
print(holo.image)
file = open(write_path + ("/img_%s.p" % output_name), 'wb')
pickle.dump(holo.image, file)
file.close()
