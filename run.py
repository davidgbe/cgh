from lib.mnist import MNIST
from lib import utilities
from lib.real_object import RealObject
import numpy as np
import multiprocessing as mp
import pickle
import sys
from random import randint as rand

mp.freeze_support()
image = MNIST.get_image()

ro = RealObject(image, np.zeros(3), 1, 1)
print(sys.argv)
holo = ro.generate_interference_pattern(np.array([4, 0, 0]), 1, 1)
print(holo.image)
file = open(sys.argv[1] + ("/img_%s.p" % sys.argv[2]), 'wb')
pickle.dump(holo.image, file)
file.close()
