from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from utils.pg_utils import PGConv2d


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_models(model, path, suffix=''):
    """Save model to given path
    Args:
        model: model to be saved
        path: path that the model would be saved
        epoch: the epoch the model finished training
    """
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, "{}.pt".format(suffix))
    torch.save(model, file_path)  # pwf file


def load_models(model, path, suffix=''):
    """Load model from given path
    Args:
        model: model to be saved
        path: path that the model would be saved
        epoch: the epoch the model finished training
    """

    file_path = os.path.join(path, "{}.pt".format(suffix))
    model.load_state_dict(torch.load(file_path))  # pwf file


def replace_conv(model, skip_layers=None, **kwargs):
    """
    Args:
        model: model to be replaced
        skip_layers: [default None] skip some layers without replacing
        kwargs: parameters related to precision gating
    """

    # List all conv layers in the model
    conv_layers = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append(n)

    for (layer_id, layer_name) in enumerate(conv_layers):
        if not skip_layers is None and layer_id in skip_layers:
            print("Skipping: ", layer_name)
            continue
        # Get the strip path of each conv layer
        name_seq = layer_name.split(".")
        # Use DFS to replace each conv model with PGConv
        parent = model
        for mkey in name_seq:
            n_parent = parent._modules[mkey]
            # Current module is a leaf node
            if len(n_parent._modules) == 0:
                # Make sure the leaf node is a convolutioan operation
                assert(isinstance(n_parent, torch.nn.Conv2d))
                print("Replacing: ", layer_name)
                parent._modules[mkey] = PGConv2d.copy_conv(n_parent, kwargs)
                del n_parent
            else:
                parent = n_parent
    return model


def poly_decay_lr(optimizer, global_steps, total_steps, base_lr, end_lr, power):
    """Sets the learning rate to be polynomially decaying"""
    lr = (base_lr - end_lr) * (1 - global_steps/total_steps) ** power + end_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
