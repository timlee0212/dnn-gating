from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision

from timm.models import create_model
from timm.models.registry import is_model
from timm.models.vision_transformer import Attention
from timm.models.layers import Mlp
from utils.pg_utils import QLinear, PGAttention

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


def replacePGModule(model,args):
    for name,subModule in model._modules.items():
        #print('module',name,'is a ',subModule,"has",len(subModule._modules),'submodules')
        if(len(subModule._modules)!=0):
            replacePGModule(subModule,args)
        if isinstance(subModule,Attention):
            #print(model._modules[name])
            attn = model._modules[name]
            pgattn = PGAttention(attn.qkv.in_features, attn.num_heads, attn.qkv.bias is not None)
            pgattn.qkv.weight.data.copy_(attn.qkv.weight)
            pgattn.qkv.bias.data.copy_(attn.qkv.bias)
            pgattn.proj.weight.data.copy_(attn.proj.weight)
            pgattn.proj.bias.data.copy_(attn.proj.bias)
            model._modules[name] = pgattn
        elif isinstance(subModule,Mlp):
            #print(model._modules[name])
            mlp = model._modules[name]
            fc1 = QLinear(mlp.fc1.in_features, mlp.fc1.out_features,mlp.fc1.bias is not None)
            fc1.weight.data.copy_(mlp.fc1.weight)
            fc1.bias.data.copy_(mlp.fc1.bias)
            mlp.fc1 = fc1
            fc2 = QLinear(mlp.fc2.in_features, mlp.fc2.out_features,mlp.fc2.bias is not None)
            fc2.weight.data.copy_(mlp.fc2.weight)
            fc2.bias.data.copy_(mlp.fc2.bias)
            mlp.fc2 = fc2










    
  
