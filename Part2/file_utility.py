'''
This module contains utility functions
'''
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import math
import json

# function to load training images

# function to preprocess training images
def process_image(image_path):

    # Process a PIL image for use in a PyTorch model
    im = Image.open(image_path)

    width, height = im.size
    targetSize = 255
    finalSize = 224

    if width < height: # if the image is a portrait, resize and crop top and bottom
        scaledHeight = math.ceil(height * (255 / width))
        im = im.resize((255, scaledHeight), Image.ANTIALIAS)
        left = (targetSize - finalSize) // 2
        upper = (scaledHeight - 224) // 2
        right = 255 - (targetSize - finalSize - left)
        lower = scaledHeight - ((scaledHeight - 224) - (scaledHeight - 224) // 2)
    else: # if the image is a landscape, resize and crop the sides
        scaledWidth = math.ceil(width * (255 / height))
        im = im.resize((scaledWidth, 255), Image.ANTIALIAS)
        left = (scaledWidth - 224) // 2
        upper = (targetSize - finalSize) // 2
        right = scaledWidth - ((scaledWidth - 224) - (scaledWidth - 224) // 2)
        lower = 255 - (targetSize - finalSize - upper)

    im = im.crop((left, upper, right, lower))

    # Convert the image to numpy
    np_image = np.array(im)
    np_image = np_image / 255.0

    im_mean = [0.485, 0.456, 0.406]
    im_std = [0.229, 0.224, 0.225]

    np_image = (np_image - im_mean) / im_std

    np_image = np.transpose(np_image)

    image = torch.from_numpy(np_image)
    image.unsqueeze_(0)
    image = image.float()

    return image



# function to convert classes to names
def load_cat_to_name_dict(file_name):
    with open(file_name, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name