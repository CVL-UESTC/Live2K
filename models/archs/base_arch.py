import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import spectral_norm


def default_init_weights(module_list, scale=1):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(basic_block, n_basic_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        n_basic_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(n_basic_blocks):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        nf (int): Number of features. Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
        sn (bool): Whether to use spectral norm. Default: False.
        n_power_iterations (int): Used in spectral norm. Default: 1.
        sn_bias (bool): Whether to apply spectral norm to bias. Default: True.

    """

    def __init__(self,
                 nf=64,
                 res_scale=1,
                 pytorch_init=False,
                 sn=False,
                 n_power_iterations=1,
                 sn_bias=True):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if sn:
            self.conv1 = spectral_norm(
                self.conv1,
                name='weight',
                n_power_iterations=n_power_iterations)
            self.conv2 = spectral_norm(
                self.conv2,
                name='weight',
                n_power_iterations=n_power_iterations)
            if sn_bias:
                self.conv1 = spectral_norm(
                    self.conv1,
                    name='bias',
                    n_power_iterations=n_power_iterations)
                self.conv2 = spectral_norm(
                    self.conv2,
                    name='bias',
                    n_power_iterations=n_power_iterations)
        self.relu = nn.ReLU(inplace=True)

        if not sn and not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=128, nf=16, n_blocks=4):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body1 = make_layer(
            ResidualBlockNoBN, n_blocks, nf=nf)
        self.body2 = make_layer(
            ResidualBlockNoBN, n_blocks, nf=nf*4)
        self.conv_last = nn.Conv2d(16*nf, out_nc, 3, 1, 1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        # initialization
        default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body1(feat)
        feat = self.pixel_unshuffle(feat)
        feat = self.body2(feat)
        feat = self.pixel_unshuffle(feat)
        feat = self.lrelu(self.conv_last(feat))

        return feat


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        
        return x1 * x2


class DepthwiseConv(nn.Module):
    def __init__(self, c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, k, s, p, groups=c, bias=True)

    def forward(self, x):
        return self.conv(x)
