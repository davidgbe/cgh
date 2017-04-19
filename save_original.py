from lib.mnist import MNIST
from lib import utilities
from lib.real_object import RealObject
import numpy as np
import multiprocessing as mp
import pickle
import sys
from random import randint as rand

image = MNIST.get_image(sys.argv[1])

utilities.save_image(image, sys.argv[1])
