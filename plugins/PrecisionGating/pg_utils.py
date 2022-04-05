from numpy import deprecate
from timm.models.layers import Mlp
from timm.models.vision_transformer import Attention
from timm.models import levit
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
    #Compatible mode using old replace function
    if kwargs['old_rp']:
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
    else:
        # List all interested layers in the model
        #Separate the layers to gaurantee the replacement order
        conv_layers = []
        attn_layers = []
        linear_layers = []

        #Special Processing
        levit_layers = []

        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                conv_layers.append(n)
            elif isinstance(m, Attention):
                attn_layers.append(n)
            elif isinstance(m, nn.Linear):
                linear_layers.append(n)
            #Porcess Special Ones
            elif isinstance(m, (levit.Attention, levit.AttentionSubsample)):
                levit_layers.append(n)
        cand_layers = conv_layers +linear_layers +attn_layers + levit_layers
        for (layer_id, layer_name) in enumerate(cand_layers):
            # Get the strip path of each conv layer
            name_seq = layer_name.split(".")
            # Use DFS to replace each conv model with PGConv
            parent = model
            for mkey in name_seq:
                n_parent = parent._modules[mkey]
                # Current module is a leaf node
                if len(n_parent._modules) == 0:
                    # Make sure the leaf node is a convolutioan operation
                    print("Replacing: ", layer_name)
                    if isinstance(n_parent, nn.Conv2d):
                        parent._modules[mkey] = PGConv2d.copyConv(n_parent, **kwargs)
                    elif isinstance(n_parent, nn.Linear):
                        parent._modules[mkey] = QLinear.copyLinear(n_parent, wbits=kwargs['wbits'], abits=kwargs['abits'])
                    elif isinstance(n_parent, levit.Attention):
                        parent._modules[mkey] = PGAttentionLeVit.copyAttn(n_parent, **kwargs)
                    elif isinstance(n_parent, levit.AttentionSubsample):
                        parent._modules[mkey] = PGAttentionLeVit.copyAttn(n_parent, **kwargs)
                    else:
                        raise ValueError("Unrecongnized Replace target {layer_name}!")
                    del n_parent
                else:
                    parent = n_parent
    return model



