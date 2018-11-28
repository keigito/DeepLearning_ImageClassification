'''
This file contains functions that are related to computing (training, predicting, processing results).
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

# A function to perform a validation
def validate(model, valid_dataloader, criterion, UseGpu):
    valid_loss = 0
    accuracy = 0
    for images, labels in valid_dataloader:
        images, labels = images.to('cuda' if UseGpu else 'cpu'), labels.to('cuda' if UseGpu else 'cpu') # Initialize the device

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

def train_model(epochs, model, train_dataloader, valid_dataloader, learning_rate, UseGpu, batch_size):
    # Set a critorion and the optimizer

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    device = torch.device('cuda' if UseGpu else 'cpu')
    model.to(device)

    steps = 0

    for e in range(epochs):
        running_loss = 0
        for i , (inputs, labels) in enumerate(train_dataloader):
            steps += 1

            inputs, labels = inputs.to('cuda' if UseGpu else 'cpu'), labels.to('cuda' if UseGpu else 'cpu')

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate train loss, validation loss validation accuracy
            if steps % batch_size == 0 and i - 393 > 0: # Perform the validation once an epoch only

                model.eval() # Turn off the training mode
                with torch.no_grad(): # Do not compute gradients to save time
                    valid_loss, accuracy = validate(model, valid_dataloader, criterion, UseGpu)

                print('Epoch {0}/{1}: Training Loss: {2}; Validation Loss: {3}, Validation Accuracy: {4}'.format(e+1, epochs, running_loss/len(train_dataloader)/batch_size, valid_loss/len(valid_dataloader), accuracy/len(valid_dataloader)))
                running_loss = 0
                model.train() # Turn on the training mode

    return model, optimizer


# function to get the top k classes
def get_top_k(result, k):
    softmax_result = F.softmax(result, dim=1)
    top_five = softmax_result.topk(k)

    return top_five