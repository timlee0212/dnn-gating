

import torch
import torch.nn as nn
from torch.autograd import Function
import math

class Quant8F(Function):

    @staticmethod
    def forward(cxt, input, dim=None, quant=8, _scale=None, _initial_zero_point=None):
        #if not isinstance(dim, int):
        #    raise NotImplemented("Currently Only Support Selecting One Dimension.")
        
        if dim!=None and input.shape[dim] < 2:
            return input

        if _scale != None:
            output = ((input/_scale + _initial_zero_point).round_().clamp_(min=0, max=(2**quant-1)) - _initial_zero_point) * _scale
        elif dim == None:
            scale = (torch.max(input) - torch.min(input)) / 255
            if scale == 0:
                #scale = 1
                return input
            initial_zero_point = 0 - torch.min(input) / scale
            if initial_zero_point < 0:
                initial_zero_point = 0
            elif initial_zero_point > 255*scale:
                initial_zero_point = 255*scale
            else:
                if math.isnan(initial_zero_point):
                    initial_zero_point = 0
            initial_zero_point = int(initial_zero_point)
            #dtype = torch.qint8
            #qm = nn.quantized.Quantize(scale, initial_zero_point, dtype)
            #dqm = nn.quantized.DeQuantize()
            #output = dqm(qm(input))
            output = ((input/scale + initial_zero_point).round_().clamp_(min=0, max=(2**quant-1)) - initial_zero_point) * scale
        else:
            scale = (1.0/(2**quant-1)) * (torch.max(input, dim=dim, keepdim=True)[0] - torch.min(input, dim=dim, keepdim=True)[0])
            if torch.count_nonzero(scale) == 0:
                return input
            scale[scale==0] = 1
            #initial_zero_point = 0 + -1*torch.min(input, dim=dim, keepdim=True)[0]
            #initial_zero_point[initial_zero_point<0] = 0
            #initial_zero_point[initial_zero_point>(2**quant-1)] = (2**quant-1)
            #initial_zero_point = 0 + -1*torch.div(initial_zero_point, scale)
            initial_zero_point = 0 + -1*torch.div(torch.min(input, dim=dim, keepdim=True)[0], scale)
            initial_zero_point[initial_zero_point<0] = 0
            initial_zero_point[initial_zero_point>(2**quant-1)] = (2**quant-1)
            initial_zero_point[initial_zero_point != initial_zero_point] = 0
            initial_zero_point = initial_zero_point.int()
            output = ((input/scale + initial_zero_point).round_().clamp_(min=0, max=(2**quant-1)) - initial_zero_point) * scale
            
        #print("SCALE = {}".format(scale))
        #print("ZERO_POINT = {}".format(zero_point))
        
        #dtype = torch.qint8
        #qm = nn.quantized.Quantize(scale, zero_point, dtype)
        #dqm = nn.quantized.DeQuantize()

        #output = dqm(qm(input))        
        #output = ((input/scale + initial_zero_point).round_().clamp_(min=0, max=(2**quant-1)) - initial_zero_point) * scale
        
        #mse_loss = nn.MSELoss()
        #loss = mse_loss(input, output)
        #print("Quantization loss: {}".format(loss))
        
        return output
        
    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


# alias
quant8 = Quant8F.apply