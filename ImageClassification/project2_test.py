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

def parse_args():
    parser = argparse.ArgumentParser(description= \
        'scipt for evaluation of project 2')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='Used when there are cuda installed.')
    parser.add_argument('--output_path', default='./', type=str,
        help='The path that stores the log files.')

    pargs = parser.parse_args()
    return pargs

def create_logger(final_output_path):
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger

def eval_net(net, loader, logging):
    net = net.eval()
    if args.cuda:
        net = net.cuda()

    # use the trained network by default
    model_name = args.output_path + 'project2.pth'

    if args.cuda:
        net.load_state_dict(torch.load(model_name, map_location='cuda'))
    else:
        net.load_state_dict(torch.load(model_name, map_location='cpu'))

    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        if args.cuda:
            outputs = outputs.cpu()
            labels = labels.cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print and write to log.
    logging.info('=' * 55)
    logging.info('SUMMARY of Project2')
    logger.info('The number of testing image is {}'.format(total))
    logging.info('Accuracy of the network on the test images: {} %'.format(100 * round(correct / total, 4)))
    logging.info('=' * 55)

# Prepare for writing logs and setting GPU. 
args = parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
# print('using args:\n', args)

logger = create_logger(args.output_path)
logger.info('using args:')
logger.info(args)

test_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# Define the test dataset and dataloader.

test_image_path = '../test'  # DO NOT CHANGE THIS LINE

testset = ImageFolder(test_image_path, test_transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

####################################

# ==================================
# test the network and write to logs. 
# use cuda if called with '--cuda'. 

# We used pre defined resnet18
network = resnet18()
if args.cuda:
    network = resnet18().cuda()

# test the trained network
eval_net(network, testloader, logging)
# ==================================
