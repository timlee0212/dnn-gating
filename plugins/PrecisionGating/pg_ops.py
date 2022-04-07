from timm.models import levit

from .pg_modules import *


class QConv2d(nn.Conv2d):
    """
    A convolutional layer with its weight tensor and input tensor quantized.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', wbits=8,abits=8):
        super(QConv2d, self).__init__(in_channels, out_channels,
                                      kernel_size, stride,
                                      padding, dilation, groups,
                                      bias, padding_mode)
        self.register_buffer('weight_fp', self.weight.data.clone())

        self.quantize_w = TorchQuantize(wbits)
        self.quantize_a = TorchQuantize(abits)

    def forward(self, input):
        """
        1. Quantize the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform convolution
        """
        return F.conv2d(self.quantize_a(input),
                        self.quantize_w(self.weight),
                        self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


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
                       abits=kwargs['abits'])

        # Replicate weight
        new_conv.weight.data = conv.weight.data.clone()
        if not new_conv.bias is None:
            new_conv.bias.data = conv.bias.data.clone()
        new_conv.weight_fp = conv.weight.data.clone()
        return new_conv



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
        super().__init__()

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
    def copyConv(cls, conv, quant_only=False, **kwargs):
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
        if quant_only:
            return new_conv.conv
        else:
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
    def copyLinear(cls, linear: torch.nn.Linear, **kwargs):
        qln = cls(linear.in_features, linear.out_features,
                  linear.bias is not None, **kwargs)
        qln.weight.data.copy_(linear.weight)
        qln.weight_fp.data.copy_(linear.weight)
        if linear.bias is not None:
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
                 sparse_bp=False, th=0.99):
        super().__init__()
        self.mask = None
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.threshold = th
        self.gt = SparseGreaterThan if sparse_bp else GreaterThan

        self.qkv = QLinear(dim, dim * 3, bias=qkv_bias,
                           wbits=wbits, abits=abits)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = QLinear(dim, dim, wbits=wbits)
        self.proj_drop = nn.Dropout(proj_drop)
        self.quantize_a = TorchQuantize(abits)
        self.quantize_MSB = TorchQuantize(pgabits)
        self.greaterThan = GreaterThan.apply

        self.num_out = 0
        """ number of output features computed at high precision """
        self.num_high = 0

    @classmethod
    def copyAttn(cls, attn, **kwargs):
        pgattn = cls(attn.qkv.in_features, attn.num_heads,
                     attn.qkv.bias is not None, **kwargs)
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
        # qkv_msb = self.quantize_MSB(qkv)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_msb = self.quantize_MSB(q)
        k_msb = self.quantize_MSB(k)
        attn_msb = (q_msb @ k_msb.transpose(-2, -1)) * self.scale
        # attn_msb = self.quantize_noise(attn_msb)
        attn_msb = attn_msb.softmax(dim=-1)
        mask = self.gt.apply(attn_msb, self.threshold)
        msb_mask = torch.zeros_like(mask)
        self.mask = mask

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn * mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.quantize_a(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        self.num_out = mask.numel()
        self.num_high = torch.sum(mask).item()
        return x

class PGAttentionPVT(PGAttention):
    def __init__(self,  dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, wbits=8, abits=8, pgabits=4,
                 sparse_bp=False, th=0.99):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop, wbits, abits, pgabits, sparse_bp, th)
        delattr(self, "qkv")
        
        self.scale = qk_scale or (dim//num_heads) ** -0.5
        self.q = QLinear(dim, dim, bias=qkv_bias, wbits=wbits, abits=abits)
        self.kv = QLinear(dim, dim * 2, bias=qkv_bias, wbits=wbits, abits=abits)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = QConv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, wbits=wbits, abits=abits)
            self.norm = nn.LayerNorm(dim)

    @classmethod
    def copyAttn(cls, attn, **kwargs):
        pgattn = cls(attn.dim, attn.num_heads,
                     attn.kv.bias is not None, attn_drop = attn.attn_drop.p,
                     proj_drop = attn.proj_drop.p, sr_ratio = attn.sr_ratio, **kwargs)
        pgattn.q = QLinear.copyLinear(attn.q, wbits=kwargs['wbits'], abits=kwargs['abits'])
        pgattn.kv = QLinear.copyLinear(attn.kv, wbits=kwargs['wbits'], abits=kwargs['abits'])
        pgattn.proj = QLinear.copyLinear(attn.proj, wbits=kwargs['wbits'])
        if attn.sr_ratio > 1:
            kwargs['th'] = 0.0
            pgattn.sr = QConv2d.copyConv(attn.sr, wbits=kwargs['wbits'], abits=kwargs['abits'] )
            pgattn.norm = attn.norm

        return pgattn


    #Override Forward Function
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.quantize_a(self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.quantize_a(self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        else:
            kv = self.quantize_a(self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        #Generate Mask
        q_msb = self.quantize_MSB(q)
        k_msb = self.quantize_MSB(k)
        attn_msb = (q_msb @ k_msb.transpose(-2, -1)) * self.scale
        # attn_msb = self.quantize_noise(attn_msb)
        attn_msb = attn_msb.softmax(dim=-1)
        self.mask = self.gt.apply(attn_msb.abs(), self.threshold)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn * self.mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.quantize_a(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        self.num_out = self.mask.numel()
        self.num_high = torch.sum(self.mask).item()

        return x


# Special OPs for LeViT
class PGAttentionLeVit(levit.Attention):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4, act_layer=None, resolution=14, use_conv=False, wbits=8,
                 abits=8, pgabits=4, sparse_bp=False, th=0.99):
        super().__init__(dim, key_dim, num_heads, attn_ratio, act_layer, resolution, use_conv)

        self.threshold = 0
        self.gt = SparseGreaterThan if sparse_bp else GreaterThan
        self.mask = None
        self.quantize_a = TorchQuantize(abits)
        self.quantize_MSB = TorchQuantize(pgabits)
        self.num_out = 0
        """ number of output features computed at high precision """
        self.num_high = 0

        # Now we override some modules
        # This is hardcoded in the levit code, Caution possible changes for furutre version!
        self.qkv.c = QConv2d.copyConv(self.qkv.c, wbits=wbits, abits=abits) \
            if self.use_conv else QLinear.copyLinear(self.qkv.c, wbits=wbits, abits=abits)
        self.proj[1].c = QConv2d.copyConv(self.proj[1].c, wbits=wbits, abits=abits) \
            if self.use_conv else QLinear.copyLinear(self.proj[1].c, wbits=wbits, abits=abits)

    @classmethod
    def copyAttn(cls, leAttn: levit.Attention, **kwargs):

        dim = leAttn.qkv.c.in_channels if leAttn.use_conv else leAttn.qkv.c.in_features
        pgattn = cls(dim, leAttn.key_dim, leAttn.num_heads, leAttn.attn_ratio,
                     leAttn.proj[0].__class__, use_conv=leAttn.use_conv, **kwargs)

        # Now we copy the weights
        pgattn.qkv.c = QConv2d.copyConv(leAttn.qkv.c, wbits=kwargs['wbits'], abits=kwargs['abits']) \
            if pgattn.use_conv else QLinear.copyLinear(leAttn.qkv.c, wbits=kwargs['wbits'], abits=kwargs['abits'])

        pgattn.proj[1].c = QConv2d.copyConv(leAttn.proj[1].c, wbits=kwargs['wbits'], abits=kwargs['abits']) \
            if pgattn.use_conv else QLinear.copyLinear(leAttn.proj[1].c, wbits=kwargs['wbits'], abits=kwargs['abits'])

        #Now we copy BN
        pgattn.qkv.bn = leAttn.qkv.bn
        pgattn.proj.bn = leAttn.proj.bn

        # Recopy the buffer
        pgattn.attention_biases = nn.Parameter(leAttn.attention_biases.clone())
        delattr(pgattn, "attention_bias_idxs")
        pgattn.register_buffer("attention_bias_idxs", leAttn.attention_bias_idxs.clone())
        pgattn.ab = leAttn.ab

        
        return pgattn

    # Override the forward function

    def forward(self, x):
        if self.use_conv:
            B, C, H, W = x.shape
            qkv = self.quantize_a(self.qkv(x).view(
                B, self.num_heads, -1, H * W)).split([self.key_dim, self.key_dim, self.d], dim=2)
            q, k, v = qkv[0], qkv[1], qkv[2]
            self.mask = self._gen_mask(q, k)

            attn = (q.transpose(-2, -1) @ k) * self.scale + \
                   self.get_attention_biases(x.device)
            attn = (attn * self.mask).softmax(dim=-1)

            attn = self.quantize_a(attn)
            x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        else:
            B, N, C = x.shape
            q, k, v = self.quantize_a(self.qkv(x).view(
                B, N, self.num_heads, -1)).split([self.key_dim, self.key_dim, self.d], dim=3)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            self.mask = self._gen_mask(q, k)

            attn = q @ k.transpose(-2, -1) * self.scale + \
                   self.get_attention_biases(x.device)
            attn = (attn * self.mask).softmax(dim=-1)

            attn = self.quantize_a(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        self.num_out = self.mask.numel()
        self.num_high = torch.sum(self.mask).item()
        return x


    def _gen_mask(self, q, k):
        q_msb = self.quantize_MSB(q)
        k_msb = self.quantize_MSB(k)
        attn_msb = (q_msb @ k_msb.transpose(-2, -1)) * self.scale
        # attn_msb = self.quantize_noise(attn_msb)
        attn_msb = attn_msb.softmax(dim=-1)
        mask = self.gt.apply(attn_msb, self.threshold)
        return mask


class PGAttentionSubsampleLeVit(levit.AttentionSubsample):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8, attn_ratio=2, act_layer=None, resolution=14,
                 resolution_=7, stride=2, use_conv=False, wbits=8, abits=8, pgabits=4, sparse_bp=False, th=0.99):
        super().__init__(
            in_dim, out_dim, key_dim, num_heads, attn_ratio, act_layer, stride, resolution, resolution_, use_conv)

        self.threshold = 0.0
        self.gt = SparseGreaterThan if sparse_bp else GreaterThan
        self.mask = None
        self.quantize_a = TorchQuantize(abits)
        self.quantize_MSB = TorchQuantize(pgabits)
        self.num_out = 0
        """ number of output features computed at high precision """
        self.num_high = 0

        # Now we override some modules
        # This is hardcoded in the levit code, Caution possible changes for furutre version!
        self.kv.c = QConv2d.copyConv(self.kv.c, wbits=wbits, abits=abits) \
            if self.use_conv else QLinear.copyLinear(self.kv.c, wbits=wbits, abits=abits)
        self.q[1].c = QConv2d.copyConv(self.q[1].c, qwbits=wbits, abits=abits) \
            if self.use_conv else QLinear.copyLinear(self.q[1].c, wbits=wbits, abits=abits)

        self.proj[1].c = QConv2d.copyConv(self.proj[1].c, wbits=wbits, abits=abits) \
            if self.use_conv else QLinear.copyLinear(self.proj[1].c, wbits=wbits, abits=abits)

    @classmethod
    def copyAttn(cls, leAttnSS: levit.AttentionSubsample, **kwargs):
        # Resolution is not used here since we already replace the biases
        # The resolution is not used in any other coponents in the model
        in_dim = leAttnSS.kv.c.in_channels if leAttnSS.use_conv else leAttnSS.kv.c.in_features
        out_dim = leAttnSS.proj[1].c.out_channels if leAttnSS.use_conv else leAttnSS.proj[1].c.out_features

        pgattn = cls(in_dim, out_dim, leAttnSS.key_dim, leAttnSS.num_heads, leAttnSS.attn_ratio,
                     leAttnSS.proj[0].__class__, resolution=leAttnSS.resolution, resolution_=leAttnSS.resolution_,
                     stride=leAttnSS.stride, use_conv=leAttnSS.use_conv, **kwargs)
        # Now we copy the weights
        pgattn.attention_biases = nn.Parameter(leAttnSS.attention_biases.clone())
        delattr(pgattn, "attention_bias_idxs")
        pgattn.register_buffer("attention_bias_idxs", leAttnSS.attention_bias_idxs.clone())
        pgattn.ab = leAttnSS.ab
        pgattn.kv.c = QConv2d.copyConv(leAttnSS.kv.c, wbits=kwargs['wbits'], abits=kwargs['abits']) \
            if pgattn.use_conv else QLinear.copyLinear(leAttnSS.kv.c, wbits=kwargs['wbits'], abits=kwargs['abits'])
        pgattn.q[1].c = QConv2d.copyConv(leAttnSS.q[1].c, wbits=kwargs['wbits'], abits=kwargs['abits']) \
            if pgattn.use_conv else QLinear.copyLinear(leAttnSS.q[1].c, wbits=kwargs['wbits'], abits=kwargs['abits'])
        pgattn.proj[1].c = QConv2d.copyConv(leAttnSS.proj[1].c, quant_only=True, wbits=kwargs['wbits'], abits=kwargs['abits']) \
            if pgattn.use_conv else QLinear.copyLinear(leAttnSS.proj[1].c, wbits=kwargs['wbits'], abits=kwargs['abits'])
        
        #Copy BN 
        pgattn.kv.bn = leAttnSS.kv.bn
        pgattn.q[1].bn = leAttnSS.q[1].bn
        pgattn.proj[1].bn = leAttnSS.proj[1].bn

        return pgattn

    # Override the forward function

    def forward(self, x):
        if self.use_conv:
            B, C, H, W = x.shape
            kv = self.quantize_a(self.kv(x).view(
                B, self.num_heads, -1, H * W)).split([self.key_dim, self.d], dim=2)
            k, v = kv[0], kv[1]
            q = self.quantize_a(self.q(x).view(
                B, self.num_heads, self.key_dim, self.resolution_2))
            self.mask = self._gen_mask(q, k)

            attn = (q.transpose(-2, -1) @ k) * self.scale + \
                   self.get_attention_biases(x.device)
            attn = (attn * self.mask).softmax(dim=-1)

            attn = self.quantize_a(attn)
            x = (v @ attn.transpose(-2, -1)).reshape(B, - \
                1, self.resolution_, self.resolution_)
        else:
            B, N, C = x.shape
            k, v = self.quantize_a(self.kv(x).view(B, N, self.num_heads, -1)).split([self.key_dim, self.d], dim=3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            q = self.quantize_a(self.q(x).view(
                B, self.resolution_2, self.num_heads, self.key_dim).permute(0, 2, 1, 3))
            self.mask = self._gen_mask(q, k)

            attn = q @ k.transpose(-2, -1) * self.scale + \
                   self.get_attention_biases(x.device)
            attn = (attn * self.mask).softmax(dim=-1)

            attn = self.quantize_a(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        self.num_out = self.mask.numel()
        self.num_high = torch.sum(self.mask).item()
        return x

    def _gen_mask(self, q, k):
        q_msb = self.quantize_MSB(q)
        k_msb = self.quantize_MSB(k)
        attn_msb = (q_msb @ k_msb.transpose(-2, -1)) * self.scale
        # attn_msb = self.quantize_noise(attn_msb)
        attn_msb = attn_msb.softmax(dim=-1)
        mask = self.gt.apply(attn_msb, self.threshold)
        return mask
