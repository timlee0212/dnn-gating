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

import numpy as np
import os, time, sys
import argparse

import utils.pg_utils as q

torch.manual_seed(123123)

#########################
# parameters 

batch_size = 128
num_epoch = 200
_LAST_EPOCH = -1 #last_epoch arg is useful for restart
_WEIGHT_DECAY = 1e-4
_ARCH = "lenet"
this_file_path = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(this_file_path, 'save_CIFAR10_model')
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
parser.add_argument('--wbits', '-w', type=int, default=0, help='bitwidth of weights')
parser.add_argument('--abits', '-a', type=int, default=0, help='bitwidth of activations')
parser.add_argument('--ispact', '-pact', action='store_true', help='activate PACT ReLU')

# PG specific arguments
parser.add_argument('--pabits', '-pab', type=int, default=4, help='a bitwidth of predictions')
parser.add_argument('--pwbits', '-pwb', type=int, default=2, help='w bitwidth of predictions')
parser.add_argument('--gtarget', '-gtar', type=float, default=0.0, help='gating target')
parser.add_argument('--sparse_bp', '-spbp', action='store_true', help='sparse backprop of PGConv2d')

parser.add_argument('--sigma', '-sg', type=float, default=0.001, help='the penalty factor')
parser.add_argument('--finetune', '-ft', action='store_true', default=False, help='finetuning')
parser.add_argument('--learningrate', '-lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--modelFt', '-model', type=str, default="self", help='finetuning name of model')
parser.add_argument('--threshold', '-th', type=float, default=0.99, help='finetuning name of model')
args = parser.parse_args()


modelName = _ARCH+"w"+str(args.wbits)+"a"+str(args.abits)
modelName += "wp"+str(args.pwbits)+"ap"+str(args.pabits)+"th"+str(args.threshold)
#----------------------------
# Load the CIFAR-10 dataset.
#----------------------------

def load_mnist():
    transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
                    transforms.ToTensor()
        ])

    # pin_memory=True makes transfering data from host to GPU faster
    trainset = torchvision.datasets.MNIST(root='/tmp/mnist_data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.MNIST(root='/tmp/mnist_data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=4, pin_memory=True)

    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')

    return trainloader, testloader, classes


#----------------------------
# Define the model.
#----------------------------

def generate_model(model_arch):
    if model_arch == 'lenet':
        from model.pg_mnist_lenet import LeNet
        return LeNet()
    else:
        raise NotImplementedError("Model architecture is not supported.")



#----------------------------
# Train the network.
#----------------------------

def train_model(trainloader, testloader, net, device):
    if torch.cuda.device_count() > 1:
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        print("Activate multi GPU support.")
        net = nn.DataParallel(net)
    net.to(device)
    # define the loss function
    criterion = (nn.CrossEntropyLoss().cuda() 
                if torch.cuda.is_available() else nn.CrossEntropyLoss())
    # Scale the lr linearly with the batch size. 
    # Should be 0.1 when batch_size=128
    initial_lr = args.learningrate * batch_size / 128
    # initialize the optimizer
    optimizer = optim.SGD(net.parameters(), 
                          lr=initial_lr, 
                          momentum=0.9,
                          weight_decay=_WEIGHT_DECAY)
    # multiply the lr by 0.1 at 100, 150, and 200 epochs
    div = num_epoch // 3
    lr_decay_milestones = [div, div*2,div*3]
    scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer, 
                        milestones=lr_decay_milestones, 
                        gamma=0.1,
                        last_epoch=_LAST_EPOCH)
    if args.finetune:
        if args.modelFt != "self":
            modelName_ = _ARCH+args.modelFt
        else:
            modelName_ = modelName
        print("Loading Pretrained Model:",modelName_)
        util.load_models(net,save_folder,suffix=modelName_)
        correctBest = test_accu(testloader, net, device)
        util.load_models(net,save_folder,suffix=modelName)
    else:
        correctBest = 0

    for epoch in range(num_epoch): # loop over the dataset multiple times

        # set printing functions
        batch_time = util.AverageMeter('Time/batch', ':.3f')
        losses = util.AverageMeter('Loss', ':6.2f')
        top1 = util.AverageMeter('Acc', ':6.2f')
        progress = util.ProgressMeter(
                        len(trainloader),
                        [losses, top1, batch_time],
                        prefix="Epoch: [{}]".format(epoch+1)
                        )

        # switch the model to the training mode
        net.train()

        print('current learning rate = {}'.format(optimizer.param_groups[0]['lr']))
        
        # each epoch
        end = time.time()
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            for name, param in net.named_parameters():
                if 'threshold' in name:
                    loss += args.sigma * torch.norm(param-args.gtarget)
            loss.backward()  
            for p in net.modules():
                if hasattr(p, 'weight_fp'):
                    p.weight.data.copy_(p.weight_fp)

            optimizer.step()
  
            for p in net.modules():
                if hasattr(p, 'weight_fp'):
                    p.weight_fp.data.copy_(p.weight.data.clamp_(-1,1))

            # measure accuracy and record loss
            _, batch_predicted = torch.max(outputs.data, 1)
            batch_accu = 100.0 * (batch_predicted == labels).sum().item() / labels.size(0)
            losses.update(loss.item(), labels.size(0))
            top1.update(batch_accu, labels.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 49:    
                # print statistics every 100 mini-batches each epoch
                progress.display(i) # i = batch id in the epoch

        # update the learning rate
        scheduler.step()

        # print test accuracy every few epochs
        print('epoch {}'.format(epoch+1))
        correct = test_accu(testloader, net, device)
        if correct > correctBest:
            print("Saving the trained model.")
            util.save_models(net.state_dict(), save_folder, suffix=modelName)
            correctBest = correct


    print('Finished Training')


#----------------------------
# Test accuracy.
#----------------------------

def test_accu(testloader, net, device):
    net.to(device)
    cnt_out = np.zeros(2) # this 9 is hardcoded for ResNet-20
    cnt_high = np.zeros(2) # this 9 is hardcoded for ResNet-20
    num_out = []
    num_high = []
    def _report_sparsity(m):
        classname = m.__class__.__name__
        if isinstance(m, q.PGConv2d):
            num_out.append(m.num_out)
            num_high.append(m.num_high)

    correct = 0
    total = 0
    # switch the model to the evaluation mode
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            net.apply(_report_sparsity)
            cnt_out += np.array(num_out)
            cnt_high += np.array(num_high)
            num_out = []
            num_high = []
    print(modelName)
    print('Accuracy of the network on the 10000 test images: %.1f %%' % (
        100 * correct / total))
    print('Sparsity of the update phase: %.1f %%' % (100-np.sum(cnt_high)*1.0/np.sum(cnt_out)*100))
    return correct


#----------------------------
# Test accuracy per class
#----------------------------

def per_class_test_accu(testloader, classes, net, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %.1f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


#----------------------------
# Main function.
#----------------------------

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Available GPUs: {}".format(torch.cuda.device_count()))

    print("Create {} model.".format(_ARCH))
    net = generate_model(_ARCH)
    print("Model Name:", modelName)

    print("Loading the data.")
    trainloader, testloader, classes = load_mnist()
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


