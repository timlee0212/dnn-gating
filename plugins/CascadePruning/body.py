import logging
from typing import Optional

from core.plugin import Plugin
from core.registry import registerPlugin

import torch
from rich import print
from rich.panel import Panel

from .csp_util import prune, replace_with_pruned
from .summary import summary
from .pruned_layers import PrunedConv, PrunedLinear
from .nm_pruned_layers import NMSparseConv, NMSparseLinear


@registerPlugin
class cascadePruning(Plugin):
    @property
    def pluginName(self):
        return "CascadePruning"

    # This class support three possible use cases, regularizer, pruning, or both.
    def __init__(self,  prune: bool, chunk_size: Optional[int] = None, prune_type: Optional[str] = None, spar_reg: Optional[str] = None, spar_coef: Optional[float] = None, q: Optional[float] = None, verbose : bool = False):
        # Sanity check for the parameters
        if prune:
            assert not (prune_type is None or chunk_size is None), "You must provide the parameters for pruning!"
        else:
            assert not (spar_reg is None or spar_coef is None), "You must provide the parameters for regularizers!"

        self.chunk_size = chunk_size
        self.prune_type = prune_type
        self.prune = prune
        self.q = q
        self.spar_reg = spar_reg
        self.spar_coef = spar_coef
        self.verbose = verbose

    def loadState(self, checkpoint_path=''):
        """
        No State to save
        """
        pass

    def saveState(self, checkpoint_path=''):
        """
        No State to load
        """
        pass

    # We use model creation hook here for loading the checkpoint after creating the model
    def modelCreationHook(self, model):
        model = replace_with_pruned(
            model, chunk_size=self.chunk_size, spar_reg=self.spar_reg)
        if self.prune:
            summary(model, self.verbose, title="Model Summary Before Pruning")
            prune(model, method=self.prune_type, q=self.q)
            summary(model, self.verbose, title="Model Summary After Pruning")

    # Copy full precision weight for update
    def preUpdateHook(self, model, inputs, targets, loss, iter_id):
        # If we are doing pruning in this run, the training will be for finetuning
        if self.prune:
            for m in model.modules():
                if isinstance(m, PrunedConv):
                    m.conv.weight.grad = m.conv.weight.grad.float() * m.mask.float()
                if isinstance(m, PrunedLinear):
                    m.linear.weight.grad = m.linear.weight.grad.float() * m.mask.float()

    def iterTailHook(self, model, inputs, targets, logger, iter_id):
        for p in model.modules():
            if hasattr(p, 'weight_fp'):
                p.weight_fp.data.copy_(p.weight.data.clamp_(-1, 1))

    def preBackwardHook(self, model, inputs, targets, loss, iter_id):
        if self.spar_reg is not None:
            reg_loss = torch.zeros_like(loss).to('cuda')
            if self.spar_reg == 'v1':
                for n, m in model.named_modules():
                    if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                        reg_loss += m.compute_group_lasso_v1()
            if self.spar_reg == 'v2':
                for n, m in model.named_modules():
                    if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                        reg_loss += m.compute_group_lasso_v2()
            if self.spar_reg == 'SSL':
                for n, m in model.named_modules():
                    if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                        reg_loss += m.compute_SSL()

        #print("Loss before reg: {}".format(loss))
        #print("Loss of reg: {}".format(reg_loss * spar_param))

            # loss is a tensor, so it is a mutable object and this side-effect will be reflected
            # Outside the scope of this function
            loss += reg_loss * self.spar_coef
