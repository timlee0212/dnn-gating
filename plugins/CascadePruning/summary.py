import torch.nn as nn
import torch.nn.functional as F
from .pruned_layers import *
from rich import print
from rich.table import Table
from rich.panel import Panel
from rich.console import Group


def summary(net, verbose=False, title=None):
    assert isinstance(net, nn.Module)

    table = Table(title=f"Model Summary: {net.__class__.__name__}")
    table.add_column("Layer ID", justify="right", style="cyan")
    table.add_column("Type", style="magenta", justify="center")
    table.add_column("Parameter", justify="left")
    table.add_column("Non-zero Parameter", justify="left")
    table.add_column("Sparsity(%)", justify="left", style="green")
    layer_id = 0
    num_total_params = 0
    num_total_nonzero_params = 0
    num_conv_params = 0
    num_conv_nonzero_params = 0
    num_linear_params = 0
    num_linear_nonzero_params = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedLinear) or isinstance(m, nn.Linear):
            if isinstance(m, PrunedLinear):
                weight = m.linear.weight.data.cpu().numpy()
            else:
                weight = m.weight.data.cpu().numpy()
            weight = weight.flatten()
            num_parameters = weight.shape[0]
            num_nonzero_parameters = (weight != 0).sum()
            sparisty = 1 - num_nonzero_parameters / num_parameters
            layer_id += 1
            table.add_row(f"{layer_id}", "Linear", f"{num_parameters}",
                          f"{num_nonzero_parameters}", f"{sparisty}")
            num_total_params += num_parameters
            num_total_nonzero_params += num_nonzero_parameters
            num_linear_params += num_parameters
            num_linear_nonzero_params += num_nonzero_parameters
        elif isinstance(m, PrunedConv) or isinstance(m, nn.Conv2d):
            if isinstance(m, PrunedConv):
                weight = m.conv.weight.data.cpu().numpy()
            else:
                weight = m.weight.data.cpu().numpy()
            weight = weight.flatten()
            num_parameters = weight.shape[0]
            num_nonzero_parameters = (weight != 0).sum()
            sparisty = 1 - num_nonzero_parameters / num_parameters
            layer_id += 1
            table.add_row(f"{layer_id}", "Convolutional", f"{num_parameters}",
                          f"{num_nonzero_parameters}", f"{sparisty}")
            num_total_params += num_parameters
            num_total_nonzero_params += num_nonzero_parameters
            num_conv_params += num_parameters
            num_conv_nonzero_params += num_nonzero_parameters
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            layer_id += 1
            table.add_row(f"{layer_id}", "BatchNorm", None, None, None)
        elif isinstance(m, nn.ReLU):
            layer_id += 1
            table.add_row(f"{layer_id}", "BatchNorm", None, None, None)

    total_sparisty = 1. - num_total_nonzero_params / max([1, num_total_params])
    conv_sparsity = 1. - num_conv_nonzero_params / max([1, num_conv_params])
    linear_sparsity = 1. - num_linear_nonzero_params / \
        max([1, num_linear_params])

    outputs = [
        f"Total nonzero parameters: {num_total_nonzero_params}",
        f"Total parameters: {num_total_params}",
        f"Conv sparsity: {conv_sparsity}",
        f"Linear sparsity: {linear_sparsity}",
        f"Total sparsity: {total_sparisty}"
    ]

    if verbose:
        outputs = [table, ]+outputs
    print(Panel.fit(Group(*outputs), title=title))
