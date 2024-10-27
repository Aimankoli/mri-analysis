from imgcmp import preprocess, compress
from PIL import Image
import numpy as np

import os

input_dir_test = os.path.join(os.getcwd(), 'Testing')
output_dir_test = os.path.join(os.getcwd(), 'Testing1')

input_dir_train = os.path.join(os.getcwd(), 'Training')
output_dir_train = os.path.join(os.getcwd(), 'Training1')

for subdirectories in os.listdir(input_dir_test)[3:]:
    output = os.path.join(output_dir_test, subdirectories)
    os.makedirs(output, exist_ok=True)
    for images in os.listdir(os.path.join(input_dir_test, subdirectories)):
        temp = preprocess(os.path.join(input_dir_test, subdirectories, images))
        im = compress(100, temp)
        Image.fromarray((im * 255).astype(np.uint8)).save(os.path.join(output, images))

for subdirectories in os.listdir(input_dir_train):
    output = os.path.join(output_dir_train, subdirectories)
    os.makedirs(output, exist_ok=True)
    for images in os.listdir(os.path.join(input_dir_train, subdirectories)):
        temp = preprocess(os.path.join(input_dir_train, subdirectories, images))
        im = compress(100, temp)
        Image.fromarray((im * 255).astype(np.uint8)).save(os.path.join(output, images))

