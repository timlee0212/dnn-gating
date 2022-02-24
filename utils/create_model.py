from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision

from timm.models import create_model


def createModel(args):
    model_arch = args.arch
    kwargs = {'wbits':args.wbits, 'abits':args.abits, \
              'sparse_bp':args.sparse_bp, \
              'pact':args.ispact,'pgabits':args.pgabits,'th':args.threshold}
    if model_arch == 'resnet20':
        return m.resnet20(**kwargs)
    elif model_arch == 'resnet18':
        model = create_model(
            args.arch,
            pretrained=args.pretrained,
            num_classes=1000
        )
        checkpoint = torch.hub.load_state_dict_from_url(
                model.default_cfg['url'], map_location='cpu', check_hash=True)
        model.load_state_dict(checkpoint)
        return model
    elif model_arch == 'deit_small_patch16_224':
        model = create_model(
            args.arch,
            pretrained=True,
            num_classes=1000,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
        checkpoint = torch.hub.load_state_dict_from_url(
                model.default_cfg['url'], map_location='cpu', check_hash=True)
        model.load_state_dict(checkpoint['model'])
        return model
    else:
        raise NotImplementedError("Model architecture is not supported.")
