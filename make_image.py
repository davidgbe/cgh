from lib import utilities
import pickle

with open(utilities.file_path(__file__, './datasets/img_offset_40_10.p'),'rb') as img_file:
  img_data = pickle.load(img_file)
  utilities.save_image(img_data, 'img_offset_40_10')
