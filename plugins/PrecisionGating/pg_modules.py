#Define necessary module for pg units

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##########
# PACT
##########


class PactClip(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, upper_bound):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.

            upper_bound   if input > upper_bound
        y = input         if 0 <= input <= upper_bound
            0             if input < 0
        """
        ctx.save_for_backward(input, upper_bound)
        return torch.clamp(input, 0, upper_bound.data)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        input, upper_bound, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_upper_bound = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > upper_bound] = 0
        grad_upper_bound[input <= upper_bound] = 0
        return grad_input, torch.sum(grad_upper_bound)


class PactReLU(nn.Module):
    def __init__(self, upper_bound=4.0):
        super(PactReLU, self).__init__()
        self.upper_bound = nn.Parameter(torch.tensor(upper_bound))

    def forward(self, input):
        return PactClip.apply(input, torch.tensor(4.0))


##########
# Mask
##########


class SparseGreaterThan(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input, torch.tensor(threshold))
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        input, threshold, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < threshold] = 0
        return grad_input, None


class GreaterThan(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input, None


##########
# Quant
##########


class Floor(torch.autograd.Function):
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
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


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


class TorchBinarize(nn.Module):
    """ Binarizes a value in the range [-1,+1] to {-1,+1} """

    def __init__(self):
        super(TorchBinarize, self).__init__()

    def forward(self, input):
        """  clip to [-1,1] """
        input = Clamp.apply(input, -1.0, 1.0)
        """ rescale to [0,1] """
        input = (input+1.0) / 2.0
        """ round to {0,1} """
        input = Round.apply(input)
        """ rescale back to {-1,1} """
        input = input*2.0 - 1.0
        return input


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


class TorchTruncate(nn.Module):
    """
    Quantize an input tensor to a b-bit fixed-point representation, and
    remain the bh most-significant bits.
        Args:
        input: Input tensor
        b:  Number of bits in the fixed-point
        bh: Number of most-significant bits remained
    """

    def __init__(self, b=8, bh=4):
        super(TorchTruncate, self).__init__()
        assert b > 0, "Cannot truncate floating-point numbers (b=0)."
        assert bh > 0, "Cannot output floating-point numbers (bh=0)."
        assert b > bh, "The number of MSBs are larger than the total bitwidth."
        self.b = b
        self.bh = bh
        self.epsilon = 1e-7

    def forward(self, input):
        """ extract the sign of each element """
        sign = torch.sign(input).detach()
        """ get the mantessa bits """
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input/scaling, 0.0, 1.0)
        """ round the mantessa bits to the required precision """
        input = Round.apply(input * (2.0**self.b-1.0))
        """ truncate the mantessa bits """
        input = Floor.apply(input / (2**(self.b-self.bh) * 1.0))
        """ rescale """
        input *= (2**(self.b-self.bh) * 1.0)
        input /= (2.0**self.b-1.0)
        return input * scaling * sign


class TorchQuantize(nn.Module):
    """
    Quantize an input tensor to the fixed-point representation.
        Args:
        input: Input tensor
        bits:  Number of bits in the fixed-point
    """

    def __init__(self, bits=0):
        super(TorchQuantize, self).__init__()
        if bits == 0:
            self.quantize = nn.Identity()
        elif bits == 1:
            self.quantize = TorchBinarize()
        elif bits == 2:
            raise NotImplementedError("Ternarize Function is not Implemented Yet!")
            #self.quantize = TorchTernarize()
        else:
            self.quantize = TorchRoundToBits(bits)

    def forward(self, input):
        return self.quantize(input)


class TorchQuantNoise(nn.Module):
    """
    Quantize an input tensor to the fixed-point representation.
        Args:
        input: Input tensor
        bits:  Number of bits in the fixed-point
    """

    def __init__(self, amp=0):
        super(TorchQuantNoise, self).__init__()
        self.amp = amp

    def forward(self, input):
        sign = torch.sign(input).detach()
        """ get the mantessa bits """
        input = torch.abs(input)
        scaling = torch.max(input).detach()
        import math
        scaling = math.ceil(math.log(scaling, 2))
        scaling = 2**scaling
        input = Clamp.apply(input/scaling, 0.0, 1.0)
        shift = 2**31
        """ round the mantessa bits to the required precision """
        outputInt = Round.apply(input*shift)*sign
        outputScale = scaling/shift
        noise = torch.randn(input.shape).to(input)*2**15
        return (outputInt+self.amp*noise)*outputScale
