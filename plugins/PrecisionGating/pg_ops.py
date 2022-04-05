import torch
import torch.nn as nn
import math

from .pg_modules import *
from timm.models import levit


class QConv2d(nn.Conv2d):
    """
    A convolutional layer with its weight tensor and input tensor quantized.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', wbits=8):
        super(QConv2d, self).__init__(in_channels, out_channels,
                                      kernel_size, stride,
                                      padding, dilation, groups,
                                      bias, padding_mode)
        self.register_buffer('weight_fp', self.weight.data.clone())

        self.quantize_w = TorchQuantize(wbits)

    def forward(self, input):
        """
        1. Quantize the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform convolution
        """
        return F.conv2d(input,
                        self.quantize_w(self.weight),
                        self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class PGConv2d(nn.Module):
    """
    A convolutional layer computed as out = out_msb + mask . out_lsb
        - out_msb = I_msb * W
        - mask = (I_msb * W)  > Delta
        - out_lsb = I_lsb * W
    out_msb calculates the prediction results.
    out_lsb is only calculated where a prediction result exceeds the threshold.

    **Note**:
        1. PG predicts with <activations>.
        2. bias must set to be False!1
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', wbits=8, abits=8, pgabits=4,
                 sparse_bp=False, threshold=0.99):
        super(PGConv2d, self).__init__()

        self.conv = QConv2d(in_channels, out_channels, kernel_size, stride,
                            padding, dilation, groups, bias, padding_mode, wbits=wbits)

        self.quantMSB = TorchQuantize(pgabits)
        self.quantIn = TorchQuantize(abits)

        self.gt = SparseGreaterThan.apply if sparse_bp else GreaterThan.apply

        self.num_out = 0
        """ number of output features computed at high precision """
        self.num_high = 0
        self.th = threshold

    def forward(self, input):
        """
        1. Truncate the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform MSB convolution
        """
        msbIn = self.quantMSB(input)
        msbOut = self.conv(msbIn)
        if self.th == 0.0:
            return msbOut

        lsbIn = self.quantIn(input) - msbIn
        lsbOut = self.conv(lsbIn)
        if self.th == 1.0:
            return msbOut + lsbOut
        """ Calculate the mask """
        mask = self.gt(torch.sigmoid(msbOut), self.th)
        """ update report """
        self.num_out = mask.numel()
        self.num_high = torch.sum(mask).item()
        return msbOut + mask * lsbOut

    @classmethod
    def copyConv(cls, conv, **kwargs):
        """
        Alternative constrtor to directly copy from the current convolutional layer
        """
        assert (conv.bias is None, "The bias of the conv must be false!")
        new_conv = cls(conv.in_channels, conv.out_channels,
                       kernel_size=conv.kernel_size,
                       dilation=conv.dilation,
                       groups=conv.groups,
                       padding_mode=conv.padding_mode,
                       stride=conv.stride,
                       padding=conv.padding,
                       bias=(not conv.bias is None),
                       wbits=kwargs['wbits'],
                       abits=kwargs['abits'],
                       pgabits=kwargs['pgabits'],
                       sparse_bp=kwargs['sparse_bp'],
                       threshold=kwargs['th'])

        # Replicate weight
        new_conv.conv.weight.data = conv.weight.data.clone()
        if not new_conv.conv.bias is None:
            new_conv.conv.bias.data = conv.bias.data.clone()
        new_conv.conv.weight_fp = conv.weight.data.clone()
        return new_conv


class QLinear(nn.Linear):
    """
    A convolutional layer with its weight tensor and input tensor quantized.
    """

    def __init__(self, in_features, out_features, bias=True, wbits=8, abits=8):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('weight_fp', self.weight.data.clone())

        self.quantize_w = TorchQuantize(wbits)
        self.quantize_a = TorchQuantize(abits)
        # self.quantize_b = TorchQuantize(32)

    @classmethod
    def copyLinear(cls, linear : torch.nn.Linear, **kwargs):
        qln = cls(linear.in_features, linear.out_features, linear.bias is not None, **kwargs)
        qln.weight.data.copy_(linear.weight)
        qln.weight_fp.data.copy_(linear.weight)
        qln.bias.data.copy_(linear.bias)
        return qln

    def forward(self, input):
        """
        1. Quantize the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform fc
        """
        return F.linear(self.quantize_a(input),
                        self.quantize_w(self.weight),
                        self.bias)


class PGAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., wbits=8, abits=8, pgabits=4,
                 sparse_bp=False, ena_schedule=False, n_banks=8, sched_th=4, th=0.99):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.n_banks = n_banks
        self.sched_th = sched_th

        self.threshold = th
        self.gt = SparseGreaterThan if sparse_bp else GreaterThan

        self.qkv = QLinear(dim, dim * 3, bias=qkv_bias,
                           wbits=wbits, abits=abits)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = QLinear(dim, dim, wbits=wbits, abits=abits)
        self.proj_drop = nn.Dropout(proj_drop)
        self.quantize_a = TorchQuantize(abits)
        self.quantize_MSB = TorchQuantize(pgabits)
        self.greaterThan = GreaterThan.apply
        self.ena_schedule = ena_schedule

        self.num_out = 0
        """ number of output features computed at high precision """
        self.num_high = 0

    @classmethod
    def copyAttn(cls, attn, **kwargs):
        pgattn = cls(attn.qkv.in_features, attn.num_heads, attn.qkv.bias is not None, **kwargs)
        pgattn.qkv.weight.data.copy_(attn.qkv.weight)
        pgattn.qkv.weight_fp.data.copy_(attn.qkv.weight)
        pgattn.qkv.bias.data.copy_(attn.qkv.bias)
        pgattn.proj.weight.data.copy_(attn.proj.weight)
        pgattn.proj.weight_fp.data.copy_(attn.proj.weight)
        pgattn.proj.bias.data.copy_(attn.proj.bias)
        return pgattn

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.quantize_a(qkv)
        #qkv_msb = self.quantize_MSB(qkv)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_msb = self.quantize_MSB(q)
        k_msb = self.quantize_MSB(k)
        attn_msb = (q_msb @ k_msb.transpose(-2, -1)) * self.scale
        # attn_msb = self.quantize_noise(attn_msb)
        attn_msb = attn_msb.softmax(dim=-1)
        mask = self.gt.apply(attn_msb, self.threshold)
        msb_mask = torch.zeros_like(mask)

        # Manipulate the mask according our scheduling scheme
        if not self.training and self.ena_schedule:
            # Attention Mask Shape [batch, head, token, token]
            sched_mask_split = torch.tensor_split(
                mask, math.ceil(mask.shape[2]//self.n_banks), dim=2)
            sched_msb_mask_split = torch.tensor_split(
                msb_mask, math.ceil(mask.shape[2]//self.n_banks), dim=2)
            # Then we get a list of VIEWS
            for (split, msb_split) in zip(sched_mask_split, sched_msb_mask_split):
                # We get a cumulative sum, we then should apply it to the non-zero position of the original tensor.
                iss_order = split.cumsum(dim=2)
                iss_order[split == 0] = 0  # Only those edge values are valid
                for idx in reversed(range(int(torch.max(iss_order).item()))):
                    #[batch, head]
                    indicator = torch.sum(iss_order == idx, dim=(2, 3))
                    sel_ = (indicator < self.sched_th).unsqueeze(2).unsqueeze(
                        3).expand(-1, -1, split.shape[2], split.shape[3])
                    sel_order_slices = iss_order[sel_]
                    sel_mask_slices = split[sel_]
                    sel_msb_mask_slices = msb_split[sel_]
                    sel_mask_slices[sel_order_slices == idx] = 0
                    sel_msb_mask_slices[sel_order_slices == idx] = 1

                    # Early Exit to reduce time, all requires schedule have been reviewed
                    if torch.sum(indicator) == sel_.shape[0] * sel_.shape[1] * split.shape[2]:
                        break

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn * mask
        if self.ena_schedule:
            attn += attn_msb * msb_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.quantize_a(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        self.num_out = mask.numel()
        self.num_high = torch.sum(mask).item()
        return x


# Special OPs for LeViT
class PGAttentionLeVit(PGAttention, levit.Attention):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4, act_layer=None, resolution=14, use_conv=False, attn_drop=0, proj_drop=0, wbits=8, abits=8, pgabits=4, sparse_bp=False, ena_schedule=False, n_banks=8, sched_th=4, th=0.99):
        PGAttention.__init__(dim, num_heads, False, attn_drop, proj_drop, wbits,
                             abits, pgabits, sparse_bp, ena_schedule, n_banks, sched_th, th)
        levit.Attention.__init__(dim, key_dim, num_heads, attn_ratio, act_layer, resolution, use_conv)

        #Now we override some modules
        #This is hardcoded in the levit code, Caution possible changes for furutre version!
        self.qkv['c'] = PGConv2d.copyConv(self.qkv['c'], wbits=wbits, abits=abits, pgabits=pgabits, sparse_bp=sparse_bp, th=th) \
        if self.use_conv else QLinear.copyLinear(self.qkv['c'], wbits=8, abits=8)
        self.proj[1]['c'] = PGConv2d.copyConv( self.proj[1]['c'], wbits=wbits, abits=abits, pgabits=pgabits, sparse_bp=sparse_bp, th=th) \
        if self.use_conv else QLinear.copyLinear( self.proj[1]['c'], wbits=8, abits=8)


    @classmethod
    def copy_attn(cls, leAttn : levit.Attention, **kwargs):
        pass
    

    #Override the forward function
    def forward(self, x):
        if self.use_conv:
            B, C, H, W = x.shape
            qkv = self.quantize_a(self.qkv(x).view(B, self.num_heads, -1, H * W)).split([self.key_dim, self.key_dim, self.d], dim=2)
            q, k, v = qkv[0], qkv[1], qkv[2]
            mask = self._gen_mask(q, k)

            attn = (q.transpose(-2, -1) @ k) * self.scale + self.get_attention_biases(x.device)
            attn = (attn * mask).softmax(dim=-1)

            x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        else:
            B, N, C = x.shape
            qkv = self.quantize_a(self.qkv(x))
            q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            mask = self._gen_mask(q, k)

            attn = q @ k.transpose(-2, -1) * self.scale + self.get_attention_biases(x.device)
            attn = (attn*mask).softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        self.num_out = mask.numel()
        self.num_high = torch.sum(mask).item()
        return x

    def _gen_mask(self, q, k):
        q_msb = self.quantize_MSB(q)
        k_msb = self.quantize_MSB(k)
        attn_msb = (q_msb @ k_msb.transpose(-2, -1)) * self.scale
        # attn_msb = self.quantize_noise(attn_msb)
        attn_msb = attn_msb.softmax(dim=-1)
        mask = self.gt.apply(attn_msb, self.threshold)
        return mask