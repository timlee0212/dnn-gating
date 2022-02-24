from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision

from timm.models import create_model
from timm.models.registry import is_model

def createModel(args):
    model_arch = args.arch
    kwargs = {'wbits':args.wbits, 'abits':args.abits, \
              'sparse_bp':args.sparse_bp, \
              'pact':args.ispact,'pgabits':args.pgabits,'th':args.threshold}
    if is_model(model_arch):
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
        if 'model' in checkpoint.keys():
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        return model    
    elif model_arch == 'resnet20':
        import model.cifar10_resnet as m 
        return m.resnet20(**kwargs)
    else:
        raise NotImplementedError("Model architecture is not supported.")
