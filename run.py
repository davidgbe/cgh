from lib.mnist import MNIST
from lib.utilities import save_image
from lib.real_object import RealObject
import numpy as np

image = MNIST.get_image()

ro = RealObject(image, np.zeros(3), 1, 1)

holo = ro.generate_interference_pattern(np.array([10, 0, 0]), 2, 2)
print(holo.image)
save_image(holo.image, 'holo')
