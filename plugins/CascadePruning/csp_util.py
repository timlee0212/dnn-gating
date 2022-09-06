import logging
import torch

from .pruned_layers import PrunedConv, PrunedLinear
from .nm_pruned_layers import NMSparseConv, NMSparseLinear

_prune_dict = {
    'percentage': "prune_by_percentage(q)",
    'std': "prune_by_std(q)",
    'dil': "prune_towards_dilation()", ########## WORKING ON IT
    'asym_dil': "prune_towards_asym_dilation()", ########## WORKING ON IT
    'sintf': "prune_structured_interfilter(q)", ########## WORKING ON IT
    'chunk': "prune_chunk(q=q)",
    'cascade': "prune_cascade_l1(q=q)",
        #m.prune_filter_chunk(q=q) ##### not a good idea to do 2-stage prune naively
    'SSL': "prune_SSL(q)",
    'cs': "prune_CambriconS(q)"
}

def prune(model, method, q=None):
    for n,m in model.named_modules():
        if isinstance(m, (PrunedConv, PrunedLinear)):
            exec(f"m.{_prune_dict[method]}")

# Replace the regular layer with the pruned layers
# We don't need the Conv2dStaticSamePadding since timm implementation use the same conv2d class
def replace_with_pruned(model, chunk_size, spar_reg):

    pruned_conv = NMSparseConv if spar_reg == "NM" else PrunedConv
    pruned_linear = NMSparseLinear if spar_reg == "NM" else PrunedLinear

    conv_layers = []
    linear_layers = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) and not isinstance(m, pruned_conv):
            conv_layers.append(n)
        elif isinstance(m, torch.nn.Linear) and not isinstance(m, pruned_linear):
            linear_layers.append(n)
    target_layers = conv_layers + linear_layers
    if len(target_layers) == 0:
        print("No Layer to replace.")
    # Now we perform the replacement
    for (layer_id, layer_name) in enumerate(target_layers):
        # Get the parent name of the module
        parent_name = ".".join(layer_name.split(".")[:-1])
        # First we check if the parent already in the candidate list to avoid duplicate the replacement process
        dup_flag = False
        for old_name in target_layers:
            if old_name in parent_name:
                dup_flag = True
                print("Layer {0} contains module {1}, skip current replacement!".format(
                    old_name, layer_name))
                break
        if dup_flag:
            continue

        # Get the legal name for sequential layer objects
        module_name = "model." + ".".join(
            [mkey if not mkey.isdigit() else ("[" + mkey + "]") for mkey in layer_name.split(".")])
        module_name = module_name.replace(".[", "[")
        if layer_name in conv_layers:
            print("Replacing CONV layer: ", layer_name)
            exec(
                "{target_module} = {module}({target_module}{args})".format(
                    target_module=module_name, module=pruned_conv.__name__, args="" if spar_reg == "NM" else ",chunk_size"))
        elif layer_name in linear_layers:
            print("Replacing Linear Layer: ", layer_name)
            exec(
                "{target_module} = {module}({target_module}{args})".format(
                    target_module=module_name, module=pruned_linear.__name__, args="" if spar_reg == "NM" else ",chunk_size"))
    return model
