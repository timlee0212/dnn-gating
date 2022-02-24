from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import utils.utils as util
from trainer.cifar10 import *

import numpy as np
import os, time, sys
import argparse

import utils.pg_utils as q

torch.manual_seed(42)

#########################
# parameters 

batch_size = 128
num_epoch = 200
_LAST_EPOCH = -1 #last_epoch arg is useful for restart
_WEIGHT_DECAY = 1e-4
this_file_path = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(this_file_path, 'cifar10_model')
#########################


#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--save', '-s', action='store_true', help='save the model')
parser.add_argument('--test', '-test', action='store_true', help='test only')
parser.add_argument('--path', '-p', type=str, default=None, help='saved model path')
parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')

# quantization
parser.add_argument('--arch', '-arch', type=str, default="resnet20", help='model architecture')
parser.add_argument('--wbits', '-w', type=int, default=0, help='bitwidth of weights')
parser.add_argument('--abits', '-a', type=int, default=0, help='bitwidth of activations')
parser.add_argument('--ispact', '-pact', action='store_true', help='activate PACT ReLU')

# PG specific arguments
parser.add_argument('--pgabits', '-pgab', type=int, default=4, help='a bitwidth of predictions')
parser.add_argument('--gtarget', '-gtar', type=float, default=0.0, help='gating target')
parser.add_argument('--sparse_bp', '-spbp', action='store_true', help='sparse backprop of PGConv2d')

parser.add_argument('--sigma', '-sg', type=float, default=0.001, help='the penalty factor')
parser.add_argument('--finetune', '-ft', action='store_true', default=False, help='finetuning')
parser.add_argument('--learningrate', '-lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--modelFt', '-model', type=str, default="self", help='finetuning name of model')
parser.add_argument('--threshold', '-th', type=float, default=0.99, help='finetuning name of model')
args = parser.parse_args()


modelName = args.arch +"_w"+str(args.wbits)+"a"+str(args.abits)
if args.threshold<1.0:
    modelName += "pga"+str(args.pgabits)+"th"+str(args.threshold)

#----------------------------
# Main function.
#----------------------------

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Available GPUs: {}".format(torch.cuda.device_count()))

    print("Create {} model.".format(args.arch))
    net = generate_model(args.arch)
    print("Model Name:", modelName)

    print("Loading the data.")
    trainloader, testloader, classes = load_cifar10()
    if args.test:
        print("Mode: Test only.")
        util.load_models(net,save_folder,suffix=modelName)
        print("Loading Model:",modelName)
        test_accu(testloader, net, device)
    else:
        print("Start training.")
        train_model(trainloader, testloader, net, device)
        test_accu(testloader, net, device)
        per_class_test_accu(testloader, classes, net, device)


if __name__ == "__main__":
    main()


