import logging
import torch
import torch.nn as nn
from core.plugin import Plugin
from core.registry import registerPlugin

class Round(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the round function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


class Clamp(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.clamp(input, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the clamp function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None


class TorchRoundToBits(nn.Module):
    """ Quantize a tensor to a bitwidth larger than 1 """

    def __init__(self, bits=2):
        super(TorchRoundToBits, self).__init__()
        assert bits > 1, "RoundToBits is only used with bitwidth larger than 1."
        self.bits = bits
        self.epsilon = 1e-7

    def forward(self, input):
        # Extract the information of quantization range, avoid leaving gradients
        with torch.no_grad():
            # The range of the full precision weights
            fp_max = torch.max(input)
            fp_min = torch.min(input)

            # The range of the quantized weight
            qmax = 2 ** self.bits - 1
            qmin = 0

            # Scaling Factor and Bias
            scale = (fp_max - fp_min) / (qmax - qmin) + self.epsilon
            bias = qmin - torch.round(fp_min / scale)
        qout = (Clamp.apply(Round.apply(input / scale + bias), qmin, qmax) - bias) * scale

        return qout


@registerPlugin
class rramNoise(Plugin):
    @property
    def pluginName(self):
        return "RRAMNoise"

    def __init__(self, high_res, low_res, bits = 8, sigma=0.02, act = False, weight = False, eval_only = False):
        self.bits = bits
        self.sigma = sigma
        self.act = act
        self.weight = weight
        self.high_cond = 1.0/high_res
        self.low_cond = 1.0/low_res
        self.eval_only = eval_only
        self.quant = TorchRoundToBits(bits)

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
        #Build a forward hook for adding noise
        def _activation_noise_hook(module, input, output):
            return self._generate_noise(output, quant_only = not self.act)

        #Add noise to weight if we only perform evaluation
        if self.eval_only:
            for n, m in model.named_parameters():
                m.data = self._generate_noise(m, quant_only = not self.weight)
                
            for n, m in model.named_modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    m.register_forward_hook(_activation_noise_hook)
            
    def evalIterHook(self, model, iter_id, logger=None):
        pass


    def evalTailHook(self, model, epoch_id=None, logger=None):
        pass

    def _generate_noise(self, val, quant_only=False):
        # Normalize the input value
        val = self.quant(val)
        if quant_only:
            return val
        fmax = torch.max(val)
        fmin = torch.min(val)
        cond = ((val-fmin) / (fmax - fmin)) * (self.high_cond - self.low_cond) + self.low_cond
        dist = torch.normal(0, self.sigma, cond.size()).to(cond.device)
        noisy = (1+dist) * cond
        #self.quant(noisy)
        val_back = ((noisy - self.low_cond) / (self.high_cond - self.low_cond)) * (fmax - fmin) + fmin
        #self.quant(res_back)
        return val_back



