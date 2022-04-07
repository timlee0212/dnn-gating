from timm.models.layers import Mlp
from timm.models.vision_transformer import Attention

from models import pvt

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
    # Compatible mode using old replace function
    if "old_rp" in kwargs and kwargs['old_rp']:
        for name, subModule in model._modules.items():
            # print('module',name,'is a ',subModule,"has",len(subModule._modules),'submodules')
            if (len(subModule._modules) != 0):
                replacePGModule(subModule, **kwargs)
            if isinstance(subModule, Attention):
                # print(model._modules[name])
                if "old_rp" in kwargs:
                    kwargs.pop("old_rp")
                attn = model._modules[name]
                model._modules[name] = PGAttention.copyAttn(attn, **kwargs)
                # Urgh....
                kwargs["old_rp"] = True
            elif isinstance(subModule, Mlp):
                # TODO: Replace all FC layers?
                # print(model._modules[name])
                if "old_rp" in kwargs:
                    kwargs.pop("old_rp")
                mlp = model._modules[name]
                mlp.fc1 = QLinear.copyLinear(mlp.fc1)
                mlp.fc2 = QLinear.copyLinear(mlp.fc2)
                # Ugly code... But we have to do this for compatbility
                kwargs["old_rp"] = True
    else:
        if "old_rp" in kwargs:
            kwargs.pop("old_rp")
        # List all interested layers in the model
        # Separate the layers to gaurantee the replacement order
        conv_layers = []
        attn_layers = []
        linear_layers = []

        # Special Processing
        levit_layers = []
        pvt_layers = []

        for n, m in model.named_modules():
            # if isinstance(m, torch.nn.Conv2d):
            #     conv_layers.append(n)
            if isinstance(m, Attention):
                attn_layers.append(n)
            elif isinstance(m, nn.Linear):
                linear_layers.append(n)
            # Porcess Special Ones
            elif isinstance(m, (levit.Attention, levit.AttentionSubsample)):
                levit_layers.append(n)
            elif isinstance(m, pvt.Attention):
                pvt_layers.append(n)
        cand_layers = conv_layers + linear_layers + attn_layers + levit_layers + pvt_layers

        for (layer_id, layer_name) in enumerate(cand_layers):
            if "head" in layer_name:
                print("Skip Classifier Head")
                continue
            # Get the parent name of the module
            parent_name = ".".join(layer_name.split(".")[:-1])
            # First we check if the parent already in the candidate list to avoid duplicate the replacement process
            dup_flag = False
            for old_name in cand_layers:
                if old_name in parent_name:
                    dup_flag = True
                    print("Layer {0} contains module {1}, skip current replacement!".format(old_name, layer_name))
                    break

            if dup_flag:
                continue

            module_name = "model." + ".".join(
                [mkey if not mkey.isdigit() else ("[" + mkey + "]") for mkey in layer_name.split(".")])
            module_name = module_name.replace(".[", "[")
            if layer_name in linear_layers:
                print("Quantizing ", layer_name)
                exec(
                    "{target_module} = QLinear.copyLinear({target_module}, wbits=kwargs['wbits'], abits=kwargs['abits'])".format(
                        target_module=module_name))
            elif layer_name in attn_layers:
                print("Replacing ", layer_name, " FOR PGAttention")
                exec(
                    "{target_module} = PGAttention.copyAttn({target_module}, **kwargs)".format(
                        target_module=module_name))
            elif layer_name in levit_layers:
                print("Replacing ", layer_name, " for PG LeViT Attention Layer")     
                exec('if isinstance({target_module}, levit.Attention):\n'
                     '   {target_module} = PGAttentionLeVit.copyAttn({target_module}, **kwargs)\n'
                     'elif isinstance({target_module}, levit.AttentionSubsample):\n'
                     '   pass#{target_module} = PGAttentionSubsampleLeVit.copyAttn({target_module}, **kwargs)'.format(
                    target_module=module_name))
            elif layer_name in pvt_layers:
                print("Replacing ", layer_name, " for PG PVT Attention Layer")
                exec(
                    "{target_module} = PGAttentionPVT.copyAttn({target_module}, **kwargs)".format(
                        target_module=module_name))
            elif layer_name in conv_layers:
                # if "blocks" in layer_name:
                #     print("Replacing ", layer_name, " for PG Conv 2D Layer")
                #     exec(
                #         "{target_module} = PGConv2d.copyConv({target_module}, **kwargs)".format(
                #             target_module=module_name))
                # else:
                print("Quantizing ", layer_name)
                exec(
                    "{target_module} = PGConv2d.copyConv({target_module}, quant_only=True, **kwargs)".format(
                        target_module=module_name))
            else:
                raise ValueError("Unrecognized Layer {0}".format(layer_name))

            # # Get the strip path of each conv layer
            # name_seq = layer_name.split(".")
            # # Use DFS to replace each conv model with PGConv
            # parent = model
            # for mkey in name_seq:
            #     n_parent = parent._modules[mkey]
            #     # Current module is a leaf node
            #     if len(n_parent._modules) == 0:
            #         # Make sure the leaf node is a convolutioan operation
            #         print("Replacing: ", layer_name)
            #         if isinstance(n_parent, nn.Conv2d):
            #             parent._modules[mkey] = PGConv2d.copyConv(n_parent, **kwargs)
            #         elif isinstance(n_parent, nn.Linear):
            #             parent._modules[mkey] = QLinear.copyLinear(n_parent, wbits=kwargs['wbits'],
            #                                                        abits=kwargs['abits'])
            #         elif isinstance(n_parent, levit.Attention):
            #             parent._modules[mkey] = PGAttentionLeVit.copyAttn(n_parent, **kwargs)
            #         elif isinstance(n_parent, levit.AttentionSubsample):
            #             parent._modules[mkey] = PGAttentionSubsampleLeVit.copyAttn(n_parent, **kwargs)
            #         else:
            #             raise ValueError("Unrecongnized Replace target {layer_name}!")
            #         del n_parent
            #     else:
            #         parent = n_parent
    return model
