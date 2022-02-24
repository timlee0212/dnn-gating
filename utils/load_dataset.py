from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms

#----------------------------
# Load the CIFAR-10 dataset.
#----------------------------

def loadCIFAR10(args):
    from timm.data import create_transform
    if args.resize > 32:
        transform_train = create_transform(
            input_size = args.resize,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(args.resize),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(args.resize, 4),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
                    transforms.ToTensor()
        ])

    # pin_memory=True makes transfering data from host to GPU faster
    trainset = torchvision.datasets.CIFAR10(root='/tmp/cifar_data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='/tmp/cifar_data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes






