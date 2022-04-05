from timm.models.layers import Mlp
from timm.models.vision_transformer import Attention

from .pg_ops import *


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
        if kwargs['skip_layers'] is not None and layer_id in kwargs['skip_layers']:
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
                assert (isinstance(n_parent, torch.nn.Conv2d))
                print("Replacing: ", layer_name)
                parent._modules[mkey] = PGConv2d.copyConv(n_parent, **kwargs)
                del n_parent
            else:
                parent = n_parent
    return model


def replacePGModule(model, **kwargs):
    for name, subModule in model._modules.items():
        # print('module',name,'is a ',subModule,"has",len(subModule._modules),'submodules')
        if (len(subModule._modules) != 0):
            replacePGModule(subModule, **kwargs)
        if isinstance(subModule, Attention):
            # print(model._modules[name])
            attn = model._modules[name]
            model._modules[name] =  PGAttention.copyAttn(attn, **kwargs)
        elif isinstance(subModule, Mlp):
            # TODO: Replace all FC layers?
            # print(model._modules[name])
            mlp = model._modules[name]
            mlp.fc1 = QLinear.copyLinear(mlp.fc1)
            mlp.fc2 = QLinear.copyLinear(mlp.fc2)