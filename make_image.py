from lib import utilities
import pickle

with open(utilities.file_path(__file__, './images/img.p'),'rb') as img_file:
  img_data = pickle.load(img_file)
  utilities.save_image(img_data, 'high_res')
