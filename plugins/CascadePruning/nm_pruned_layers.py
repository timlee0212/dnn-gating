import torch
from torch import autograd, nn
import torch.nn.functional as F


class Sparse(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b


    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None
class Sparse_NHWC(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):

        ctx.save_for_backward(weight)
        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
        w_b = w_b.permute(0,3,1,2)

        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b

    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None

class NMSparseConv(nn.Conv2d):
    """" implement N:M sparse convolution layer """
    
    def __init__(self, conv_module, bias=True, padding_mode='zeros', N=2, M=4, **kwargs):
        self.N = N
        self.M = M
        super(NMSparseConv, self).__init__(conv_module.in_channels, conv_module.out_channels, conv_module.kernel_size, conv_module.stride, conv_module.padding, conv_module.dilation, conv_module.groups, bias, padding_mode, **kwargs)
        self.weight = conv_module.weight


    def get_sparse_weights(self):

        return Sparse_NHWC.apply(self.weight, self.N, self.M)



    def forward(self, x):

        w = self.get_sparse_weights()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x



class NMSparseLinear(nn.Linear):

    def __init__(self, lin_module, bias=True, N=2, M=2, decay = 0.0002, **kwargs):
        self.N = N
        self.M = M
        super(NMSparseLinear, self).__init__(lin_module.in_features, lin_module.out_features, bias)
        self.weight = lin_module.weight


    def get_sparse_weights(self):

        return Sparse.apply(self.weight, self.N, self.M)



    def forward(self, x):

        w = self.get_sparse_weights()
        x = F.linear(x, w)
        return x


    

    

