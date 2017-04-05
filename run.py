from lib.mnist import MNIST
from lib.utilities import save_image
from lib.real_object import RealObject
import numpy as np
import multiprocessing as mp

mp.freeze_support()
image = MNIST.get_image()

save_image(image, 'original')

ro = RealObject(image, np.zeros(3), 1, 1)

holo = ro.generate_interference_pattern(np.array([4, 0, 0]), 1, 1)
print(holo.image)
save_image(holo.image, 'holo')
