
'''
This file will train and save a model
Basic usage: python train.py datadirectory
options
--save_dir str specifies the directory where the checkpoint will be saved
--arch str specifies which pretrained model to use
--hidden_units list: number of hidden units
--learning_rate  float: learning rate
--epochs int: number of epochs
--gpu binary: use GPU
'''

# Import libraries
import torch
from torchvision import datasets, transforms, models
import argparse
import model_utility
import compute_utility

# Set up arguments
parser = argparse.ArgumentParser(description='Arguments for training the image classifier using CNN.')
parser.add_argument('data_dir', type=str, help='A path to the directory where the data (image files) are stored.')
parser.add_argument('--save_dir', type=str, dest='save_dir', default='./', help='A path to the directory where the checkpoint will be saved.')
parser.add_argument('--arch', type=str, dest='arch', default='vgg11', help='The name of the pre-trained model to be used in the transfer learning. Currently limited to alexnet, vgg and densenet variants only.')
parser.add_argument('--hidden_units', type=int, dest='hidden_units', nargs='+', help='A list of the numbers of hidden layers to be used. Example: "--hidden_units 5018 1254 313" will create three hidden layers.')
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.001, help='A learning rate to be used.')
parser.add_argument('--epochs', type=int, dest='epochs', default=10, help='The number of epochs to be used for training.')
parser.add_argument('--gpu', dest='gpu', default=False, action='store_true', help='Use GPU for computations.')

args = parser.parse_args()

# Load training images (utility)
data_dir = args.data_dir # Set the top directory where the data is stored
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

batch_size = 16 # Set the batch size for the loaders
output_size = 102 # Set the number of CNN outputs to the number of flower classes

train_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ])

valid_test_transform = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                          ])

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transform)


# Using the image datasets and the trainforms, define the dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)


# Load a CNN model (model_utility)
model_name = args.arch.lower() # Get the name of the pre-trained model to use
model, input_size = model_utility.load_pretrained_model(model_name)

# Trun off the gradients for the pre-trained portion of the model
for param in model.parameters():
    param.requires_grad = False

# Build a classifier (model_utility)
hidden_layer_list = args.hidden_units # Get the number of inputs for hidden layers

classifier = model_utility.build_classifier(input_size, output_size, hidden_layer_list) # and build the classifier

model.classifier = classifier # and assign the user-defined classifier to the model

# Train the model
# Set epochs
learning_rate = args.learning_rate
epochs = args.epochs
UseGpu = args.gpu

model, optimizer = compute_utility.train_model(epochs, model, train_dataloader, valid_dataloader, learning_rate, UseGpu, batch_size)

# Create a checkpoint file (models)
file_name = 'checkpoint.pth'
save_dir = args.save_dir
class_to_idx = train_dataset.class_to_idx
full_path = model_utility.save_model(model, file_name, save_dir, epochs, optimizer, class_to_idx)

print('Model successfully saved.')
# Save the checkpoint file (models)


