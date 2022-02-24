from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import utils.utils as util
from utils.load_dataset import loadCIFAR10
from utils.create_model import createModel

import numpy as np
import os, time, sys
import argparse

torch.manual_seed(42)

#########################
# parameters 
this_file_path = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(this_file_path, 'cifar10_model')
#########################


#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training Script')
parser.add_argument('--save', action='store_true', help='save the model')
parser.add_argument('--test', action='store_true', help='test only')
parser.add_argument('--path', type=str, default=None, help='saved model path')
parser.add_argument('--gpu', type=str, default='0', help='which gpus to use')

# Training
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--epochs', type=int, default=300, help='epoch')
parser.add_argument('--pretrained', action='store_true', help='pretrained')

# Model
parser.add_argument('--resize', type=int, default=32, help='resize_image')
parser.add_argument('--arch', type=str, default="resnet20", help='model architecture')
parser.add_argument('--wbits', type=int, default=0, help='bitwidth of weights')
parser.add_argument('--abits', type=int, default=0, help='bitwidth of activations')
parser.add_argument('--ispact', action='store_true', help='activate PACT ReLU')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
parser.add_argument('--dense', action='store_true', default=False, help='is dense')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',help='Optimizer (default: "adamw"')
parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.05, help='saved model path')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
parser.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)') 
parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--decay_epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience_epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay_rate', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')

# Augmentation parameters
parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
parser.add_argument('--repeated-aug', action='store_true')
parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
parser.set_defaults(repeated_aug=True)

   # * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix_minmax',type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')


# PG specific arguments
parser.add_argument('--pgabits', type=int, default=4, help='a bitwidth of predictions')
parser.add_argument('--gtarget', type=float, default=0.0, help='gating target')
parser.add_argument('--sparse_bp', action='store_true', help='sparse backprop of PGConv2d')

parser.add_argument('--sigma', type=float, default=0.001, help='the penalty factor')
parser.add_argument('--finetune', action='store_true', default=False, help='finetuning')
parser.add_argument('--modelFt', type=str, default="self", help='finetuning name of model')
parser.add_argument('--threshold', type=float, default=0.99, help='finetuning name of model')
args = parser.parse_args()


modelName = args.arch +"_w"+str(args.wbits)+"a"+str(args.abits)
if not args.dense:
    modelName += "pga"+str(args.pgabits)+"th"+str(args.threshold)



#----------------------------
# Train the network.
#----------------------------

def train_model(trainloader, testloader, net, device):

    loss_scaler = torch.cuda.amp.GradScaler()

    from timm.optim import create_optimizer
    optimizer = create_optimizer(args, net)

    from timm.scheduler import create_scheduler
    lr_scheduler, _ = create_scheduler(args, optimizer)

    from timm.loss import SoftTargetCrossEntropy
    criterion = SoftTargetCrossEntropy()

    from timm.data import Mixup
    mixup_fn = Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes = 10)
    
    correctBest = 0

    for epoch in range(args.epochs): # loop over the dataset multiple times
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
        print(epoch)

        print('current learning rate = {}'.format(optimizer.param_groups[0]['lr']))
        # each epoch
        end = time.time()

        # update the learning rate
        lr_scheduler.step(epoch)

        for i, data in enumerate(trainloader):
            with torch.cuda.amp.autocast():
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs, labels = mixup_fn(inputs, labels)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                loss = criterion(outputs, labels)
            
            losses.update(loss.item(), labels.size(0))
            loss_scaler.scale(loss).backward()
            #for p in net.modules():
            #    if hasattr(p, 'weight_fp'):
            #        p.weight.data.copy_(p.weight_fp)

            loss_scaler.step(optimizer)
            loss_scaler.update()
  
            #for p in net.modules():
            #    if hasattr(p, 'weight_fp'):
            #        p.weight_fp.data.copy_(p.weight.data.clamp_(-1,1))


            # measure accuracy and record loss
            _, batch_predicted = torch.max(outputs.data, 1)
            _, batch_labels    = torch.max(labels.data, 1)
            batch_accu = 100.0 * (batch_predicted == batch_labels).sum().item() / labels.size(0)
            top1.update(batch_accu, labels.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 9:    
                # print statistics every 100 mini-batches each epoch
                progress.display(i) # i = batch id in the epoch


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
    #cnt_out = np.zeros(21) # this 9 is hardcoded for ResNet-20
    #cnt_high = np.zeros(21) # this 9 is hardcoded for ResNet-20
    #num_out = []
    #num_high = []
    #def _report_sparsity(m):
    #    classname = m.__class__.__name__
    #    if isinstance(m, q.PGConv2d):
    #        num_out.append(m.num_out)
    #        num_high.append(m.num_high)

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
            
            #net.apply(_report_sparsity)
            #cnt_out += np.array(num_out)
            #cnt_high += np.array(num_high)
            #num_out = []
            #num_high = []
    print(modelName)
    print('Accuracy of the network on the 10000 test images: %.1f %%' % (
        100 * correct / total))
    #print('Sparsity of the update phase: %.1f %%' % (100-np.sum(cnt_high)*1.0/np.sum(cnt_out)*100))
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load datasets
    trainloader, testloader, classes = loadCIFAR10(args)

    # create model
    #print("Create {} model.".format(args.arch))
    net = createModel(args)
    #print("Model Name:", modelName)
    

    if args.pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            net.default_cfg['url'], map_location='cpu', check_hash=True)
        net.load_state_dict(checkpoint['model'])
        net.head = torch.nn.Linear(net.num_features, 10)

    net.to(device)
    if args.test:
        print("Mode: Evalulation only.")
        #util.load_models(net,save_folder,suffix=modelName)
        print("Loading Model:",modelName)
        test_accu(testloader, net, device)
        per_class_test_accu(testloader, classes, net, device)
    else:
        print("Start training.")
        train_model(trainloader, testloader, net, device)
        test_accu(testloader, net, device)
        per_class_test_accu(testloader, classes, net, device)


if __name__ == "__main__":
    main()


