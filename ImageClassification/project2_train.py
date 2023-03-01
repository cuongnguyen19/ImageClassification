# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torchvision.models import *

from network import Network # the network we use

parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
args = parser.parse_args()

def cal_precision(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images = data[0].cuda()
            labels = data[1].cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100

# training process. 
def train_net(net, trainloader, valloader):
    val_accuracy = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # val_accuracy is the validation accuracy of each epoch.
    net = net.train()
    for epoch in range(100):  # loop over the dataset multiple times

        total = 0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs // and add GPU mode
            inputs = data[0].cuda()
            labels = data[1].cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #Calculate Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # End of Epoch
        print("Finished Epoch %d", epoch+1)

    print('Finished Training')
    val_accuracy = cal_precision(net, valloader)
    # save network
    # torch.save(net.state_dict(), args.output_path + 'project2.pth')

    return val_accuracy

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomRotation(90),
    transforms.ToTensor()
])

# Define the training dataset and dataloader.

train_image_path = '../train/' 
validation_image_path = '../validation/' 

trainset = ImageFolder(train_image_path, train_transform, target_transform=None)
valset = ImageFolder(validation_image_path, train_transform, target_transform=None)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                         shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=1,
                                         shuffle=True, num_workers=2)
####################################

# ==================================
# use cuda if called with '--cuda'.

# We used pre defined resnet18
network = resnet18()
if args.cuda:
    network = resnet18().cuda()

# train and eval the trained network

val_acc = train_net(network, trainloader, valloader)

print("final validation accuracy:", val_acc)

# ==================================
