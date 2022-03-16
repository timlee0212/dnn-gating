from .pg_ops import *
from timm.models.vision_transformer import Attention
from timm.models.layers import Mlp

def replaceConv(model, **kwargs):
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
        if 'skip_layers' in kwargs.keys() and layer_id in kwargs['skip_layers']:
            print("Skipping: ", layer_name)
            continue
        # Get the strip path of each conv layer
        name_seq = layer_name.split(".")
        # Use DFS to replace each conv model with PGConv
        parent = model
        for mkey in name_seq:
            n_parent = parent._modules[mkey]
            # Current module is a leaf nod
            if len(n_parent._modules) == 0:
                # Make sure the leaf node is a convolutioan operation
                assert(isinstance(n_parent, torch.nn.Conv2d))
                print("Replacing: ", layer_name)
                parent._modules[mkey] = PGConv2d.copy_conv(n_parent, **kwargs)
                del n_parent
            else:
                parent = n_parent
    return model

def replacePGModule(model, **kwargs):
    for name,subModule in model._modules.items():
        #print('module',name,'is a ',subModule,"has",len(subModule._modules),'submodules')
        if(len(subModule._modules)!=0):
            replacePGModule(subModule)
        if isinstance(subModule,Attention):
            #print(model._modules[name])
            attn = model._modules[name]
            pgattn = PGAttention(attn.qkv.in_features, attn.num_heads, attn.qkv.bias is not None, **kwargs)
            pgattn.qkv.weight.data.copy_(attn.qkv.weight)
            pgattn.qkv.weight_fp.data.copy_(attn.qkv.weight)
            pgattn.qkv.bias.data.copy_(attn.qkv.bias)
            pgattn.proj.weight.data.copy_(attn.proj.weight)
            pgattn.proj.weight_fp.data.copy_(attn.proj.weight)
            pgattn.proj.bias.data.copy_(attn.proj.bias)
            model._modules[name] = pgattn
        elif isinstance(subModule,Mlp):
            #print(model._modules[name])
            mlp = model._modules[name]
            fc1 = QLinear(mlp.fc1.in_features, mlp.fc1.out_features,mlp.fc1.bias is not None, kwargs['wbits'], kwargs['abits'])
            fc1.weight.data.copy_(mlp.fc1.weight)
            fc1.weight_fp.data.copy_(mlp.fc1.weight)
            fc1.bias.data.copy_(mlp.fc1.bias)
            mlp.fc1 = fc1
            fc2 = QLinear(mlp.fc2.in_features, mlp.fc2.out_features,mlp.fc2.bias is not None, kwargs['wbits'], kwargs['abits'])
            fc2.weight.data.copy_(mlp.fc2.weight)
            fc2.weight_fp.data.copy_(mlp.fc2.weight)
            fc2.bias.data.copy_(mlp.fc2.bias)
            mlp.fc2 = fc2