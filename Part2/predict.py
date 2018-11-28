
'''
This script will predict the species of a flower in a given picture
Basic usage: predict.py input checkpoint
Options:
--top_k 3: return top K most likely classes
--category_names cat_to_name.json: Use a mapping of categories to real names
--gpu: use gpu
'''

# Import libraries
from collections import OrderedDict
import numpy as np
import time
import torch
from torch import nn
from torch import optim #(not necessary?)
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import math
import argparse
import model_utility
import file_utility
import compute_utility

# Set arguments
parser = argparse.ArgumentParser(description='Arguments for classifiying the given image.')
parser.add_argument('input', type=str, metavar='', help='A path to the image file to be classified.')
parser.add_argument('checkpoint', type=str, metavar='', help='A path to the checkpoint file where the model parameters are stored.')
parser.add_argument('--top_k', type=int, dest='top_k', default=5, metavar='', help='Number of candidates to display according to the probabilities.')
parser.add_argument('--category_names', dest='category_names', type=str, default='../cat_to_name.json', metavar='', help='The file name of the category to class name dictionary to be used for mapping.')
parser.add_argument('--gpu', dest='gpu', default=False, action='store_true', help='Use GPU for computations.')

args = parser.parse_args()

# Load a saved model (model_utility)
model, class_to_idx = model_utility.load_saved_model(args.checkpoint)

# Process an image (utility)
image = file_utility.process_image(args.input)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Predict
image = image.to(device)
result = model(image)

# Get the top k classes (utility)
k = args.top_k
top_k_results = compute_utility.get_top_k(result, k)

# Convert from classes to names (utility)
likelihoods = []
names = []
if args.category_names != None:
    cat_to_name = file_utility.load_cat_to_name_dict(args.category_names)

    idx_to_class = dict((v, k) for k, v in class_to_idx.items()) # Invert the key/value pair in the class_to_idx dictionary

    idxs = []
    for i in range(k):
        likelihoods.append(float(top_k_results[0][0][i])) # Put the probabilities into a list
        idxs.append(int(top_k_results[1][0][i])) # and the corresponding indices to a list

    for v in idxs:
        cls = idx_to_class[v] # and get the classes
        name = cat_to_name[str(cls)] # and finally to flower names
        names.append(name)
else: # if no name mapping dictionary file is specified, return the class
    for i in range(k):
        names.append(top_k_results[1][0][i])

# Return the result
print('****Results****')
print(' Ranking | Name                     | Probability')
for i in range(len(names)):
    name = names[i]
    name_len = len(name)
    if name_len < 24:
        add_space = 24 - name_len
        for ii in range(add_space):
            name = name + ' '
    print('    {0}    | {1} | {2:.2f} %'.format(i+1, name, likelihoods[i] * 100.0))