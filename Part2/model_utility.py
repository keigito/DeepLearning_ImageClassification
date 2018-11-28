'''
This module contains functions related to building, saving, and loading CNN models
'''

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# A network object used to build the classifier network.
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_percent):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_percent)

    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)


# function for loading a pre-trained model. Requires a model name
def load_pretrained_model(model_name):
    try:
        if model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            input_size = 9216
        elif model_name == 'vgg11':
            model = models.vgg11(pretrained=True)
            input_size = 25088
        elif model_name == 'vgg13':
            model = models.vgg13(pretrained=True)
            input_size = 25088
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            input_size = 25088
        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=True)
            input_size = 25088
        elif model_name == 'densenet161':
            model = models.densenet161(pretrained=True)
            input_size = 2208
        elif model_name == 'densenet169':
            model = models.densenet169(pretrained=True)
            input_size = 1664
        elif model_name == 'densenet201':
            model = models.densenet201(pretrained=True)
            input_size = 1920
        else:
            raise NameError('The model name is not currently supported or not valid.') # If the name provided doesn't match, raise an error.
    except NameError() as err:
        print(err)

    return model, input_size


# function for building a classifier network. Requires the input size, output size, and hidden layer specs.
def build_classifier(input_size, output_size, hidden_layer_list):
    classifier = Network(input_size, output_size, hidden_layer_list, drop_percent=0.4) # Drop percent set to 0.4 for better performance

    return classifier


# function for saving a model
def save_model(model, name, chk_path, epochs, optimizer, class_to_idx):

    if chk_path == '': # If the path is not given, set it to the current directory
        full_name = './' + name
    else:
        full_name = chk_path + '/' + name

    model.class_to_idx = class_to_idx

    checkpoint = {'model': model,
                  'epochs': int(epochs) + 1,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_index': model.class_to_idx
                  }

    torch.save(checkpoint, full_name)

    return full_name


# function for loading a model
def load_saved_model(chk_path):
    checkpoint = torch.load(chk_path)
    model = checkpoint['model']
    class_to_idx = checkpoint['class_to_index']
    return model, class_to_idx