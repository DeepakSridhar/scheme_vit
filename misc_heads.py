import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.layers import DropPath, trunc_normal_

import pickle
import math
import numbers


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size=1, alpha=3, beta=1, padding=0, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.weight_var = nn.Parameter(torch.zeros(dim),requires_grad=True).cuda()
        self.var = self.weight_var * alpha + beta
        sigma = self.var
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        # if isinstance(sigma, numbers.Number):
            # sigma = [sigma] * dim
        
        self.pad = nn.ReflectionPad2d(padding)

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = self.pad(input)
        return self.conv(input, weight=self.weight, groups=self.groups)


# smoothing = GaussianSmoothing(3, 5, 1)
# input = torch.rand(1, 3, 100, 100)
# input = F.pad(input, (2, 2, 2, 2), mode='reflect')
# output = smoothing(input)


class DySampling(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        # self.fc2 = nn.Conv2d(out_features, out_features, 1)
        self.fc2 = DYModule(hidden_features, out_features, fc_squeeze=fc_squeeze)
        self.out_features = out_features
        # self.attention=AttentionSigmoid(in_planes=in_features,ratio=4,K=hidden_features,temprature=30,init_weight=True)
        self.drop = nn.Dropout(drop)
        self.avg = nn.AdaptiveAvgPool1d(out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    @staticmethod
    def batched_index_select(input, dim, index):
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, x):
        # soft_weights = self.attention(x).view(x.size(0), -1, 1, 1)

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        # route_prob_max, routes = torch.topk(soft_weights, self.out_features, -1)
        # route_prob_max, routes = torch.max(soft_weights, dim=-1)

        # # Get indexes of tokens going to each expert
        # indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.out_features)]
        x = self.fc1(x)
        x = self.act(x)
        sh = x.shape
        # x = self.avg((x*soft_weights).view(sh[0], -1, sh[1])).view(sh[0], -1, sh[2], sh[3])
        # x = self.batched_index_select(x, 1, routes)
        # x  = torch.index_select(x, dim=1, index=routes)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DySubSampling(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., ratio=4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.ratio = ratio
        # self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features // ratio, out_features, 1)
        # self.fc2 = DYModule(hidden_features, out_features, fc_squeeze=fc_squeeze)
        self.out_features = out_features
        self.gct1 = GCTAttention(hidden_features)
        self.drop = nn.Dropout(drop)
        # self.avg = nn.AdaptiveAvgPool1d(out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    @staticmethod
    def batched_index_select(input, dim, index):
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        sh = x.shape
        soft_weights = self.gct1(x)
        x = x*soft_weights
        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        route_prob_max, routes = torch.topk(soft_weights.view(x.size(0), -1), self.hidden_features, 1)
        y1 = self.batched_index_select(x, 1, routes[:, :self.hidden_features//self.ratio-1])
        y2 = self.batched_index_select(x, 1, routes[:, self.hidden_features//self.ratio-1:])
        x = torch.cat((y1, y2.sum(1,keepdim=True)), 1)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class ScaleTensor(nn.Module):
    def __init__(self, dim, alpha=1, beta=0, bias=True):
        super().__init__()
        self.use_bias = bias
        self.weight_ = nn.Parameter(torch.ones(dim),requires_grad=True).cuda()
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim),requires_grad=True).cuda()
        self.weight = self.weight_ * alpha + beta 
    
    def forward(self, x):
        if self.use_bias:
            return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        else:
            return x * self.weight.view(1, -1, 1, 1)


class DyMlpTokens(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,  token_mixer=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8, use_qkv=False):
        super().__init__()
        self.use_qkv = use_qkv
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if self.use_qkv:
            self.fc1 = DYModule(in_features, in_features, fc_squeeze=fc_squeeze)
        else:
            self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.act = act_layer()
        if self.use_qkv:
            self.fc2 = nn.Conv2d(in_features * 2, out_features, 1)
        else:
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        # self.token_mixer = token_mixer
        if self.use_qkv:
            # self.scale_q = ScaleTensor(in_features)
            # self.scale_k = ScaleTensor(in_features)
            self.scale_v = ScaleTensor(in_features)

        self.act2 = act_layer() #nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xorg = x
        x = self.fc1(x)
        if self.use_qkv:
            # x = torch.cat((x, self.act2(self.scale_q(self.token_mixer.q.reshape(x.shape))), self.act2(self.scale_k(self.token_mixer.k.reshape(x.shape))), self.act2(self.scale_v(self.token_mixer.v.reshape(x.shape)))), 1)
            x = torch.cat((x, self.act2(self.scale_v(xorg))), 1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class sMLPBlock(nn.Module):
    def __init__(self, in_features, W=56, H=56, hidden_features=None,  token_mixer=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        assert W == H
        channels = in_features
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels)
        self.proj_h = DYModule(H, H, fc_squeeze=fc_squeeze) #nn.Conv2d(H, H, (1, 1))
        self.proh_w = DYModule(W, W, fc_squeeze=fc_squeeze) #nn.Conv2d(W, W, (1, 1))
        self.fuse = DYModule(channels * 3, channels, fc_squeeze=fc_squeeze) #nn.Conv2d(channels*3, channels, (1,1), (1,1), bias=False)

    def forward(self, x):
        x = self.activation(self.BN(x))
        x_h = self.proj_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.proh_w(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x = self.fuse(torch.cat([x, x_h, x_w], dim=1))
        return x


class DyRatioMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        # self.fc1a = DYModule(in_features, hidden_features // 2, fc_squeeze=fc_squeeze)
        self.fc1b = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2a = nn.Conv2d(hidden_features, out_features // 2, 1)
        self.fc2b = DYModule(hidden_features, out_features // 2, fc_squeeze=fc_squeeze)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # xa = self.fc1a(x)
        x = self.fc1b(x)
        # x = torch.cat((xa, xb), 1)
        x = self.act(x)
        x = self.drop(x)
        xa = self.fc2a(x)
        xb = self.fc2b(x)
        x = torch.cat((xa, xb), 1)
        x = self.drop(x)
        return x


class GaussMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.fc1 = DYModule(in_features, in_features, fc_squeeze=fc_squeeze)
        self.mean=nn.Parameter(torch.randn(1, 1),requires_grad=True)
        self.var=nn.Parameter(torch.ones(1),requires_grad=True)
        self.mean2=nn.Parameter(torch.randn(1, 1),requires_grad=True)
        self.var2=nn.Parameter(torch.ones(1),requires_grad=True)
        self.mean3=nn.Parameter(torch.randn(1, 1),requires_grad=True)
        self.var3=nn.Parameter(torch.ones(1),requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    
    def gauss(self, x, mean, var, eps=1e-6):
        shape = x.shape
        x = x.view(-1, 1)
        # sigma = torch.diag(var)
        # inv_sigma = torch.diag(1/var)
        d = (x-mean)**2
        # print(d.shape)
        # u = torch.matmul(torch.matmul((d).unsqueeze(1), inv_sigma), (d).unsqueeze(-1))
        # g = (1/(torch.sqrt((2*np.pi)**(self.in_features)*torch.prod(var))))*torch.exp(-0.5*d/var)
        g = (1/(torch.sqrt((2*np.pi)*(var+eps))))*torch.exp(-0.5*d/(var+eps))
        g = g.view(shape)
        return g
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.gauss(x, self.mean, self.var)
        x2 = self.gauss(x, self.mean2, self.var2)
        x3 = self.gauss(x, self.mean3, self.var3)
        x = torch.cat([x, x1, x2, x3], 1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DyOREPAMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1 = DyOREPA_kxk(in_features, hidden_features, 1)#, deploy=True)
        self.act = act_layer()
        self.fc2 = DyOREPA_kxk(hidden_features, out_features, 1)#, deploy=True)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixStaticOREPAMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1 = StaticOREPA_1x1(in_features, hidden_features, 1)#, deploy=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)#, deploy=True)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixSDyOREPAMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1 = SDyOREPA_1x1(in_features, hidden_features, 1, deploy=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)#, deploy=True)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CStaticOREPAMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1 = CStaticOREPA_1x1(in_features, hidden_features, 1)#, deploy=True)
        self.act = act_layer()
        self.fc2 = CStaticOREPA_1x1(hidden_features, out_features, 1)#, deploy=True)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StaticOREPAMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1 = StaticOREPA_1x1(in_features, hidden_features, 1)#, deploy=True)
        self.act = act_layer()
        self.fc2 = StaticOREPA_1x1(hidden_features, out_features, 1)#, deploy=True)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SingleStaticOREPAMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1 = SingleStaticOREPA_1x1(in_features, hidden_features, 1)#, deploy=True)
        self.act = act_layer()
        self.fc2 = SingleStaticOREPA_1x1(hidden_features, out_features, 1)#, deploy=True)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TDyOREPAMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1 = TDyOREPA_1x1(in_features, hidden_features, 1)#, deploy=True)
        self.act = act_layer()
        self.fc2 = TDyOREPA_1x1(hidden_features, out_features, 1)#, deploy=True)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SDyOREPAMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1 = SDyOREPA_1x1(in_features, hidden_features, 1)#, deploy=True)
        self.act = act_layer()
        self.fc2 = SDyOREPA_1x1(hidden_features, out_features, 1)#, deploy=True)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OREPAMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1 = OREPA_1x1(in_features, hidden_features, 1)#, deploy=True)
        self.act = act_layer()
        self.fc2 = OREPA_1x1(hidden_features, out_features, 1)#, deploy=True)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DyMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MlpConv3d(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(1, in_features // 16, (3, 1, 1), padding=(1, 0, 0))
        self.act = act_layer()
        self.fc2 = nn.Conv3d(in_features // 16, 1, (3, 1, 1), padding=(1, 0, 0))
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        x = x.squeeze(1)
        return x




class DyMlpSmall(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = DYModuleSmall(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwitchMixConvDy(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
     
        self.fc1a = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.fc1b = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()

        # self.switch = SwitchDyConvBN(expert1=self.fc1a, expert2=self.fc1b)
        # Routing layer and softmax
        self.switch = nn.Linear(in_features, 2)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=-1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        # x = self.switch(x)
        # x = self.act(x)
        x1 = self.fc1a(x)
        x2 = self.fc1b(x)
        x1 = self.act(x1)
        x2 = self.act(x2)

        route_prob = self.softmax(self.switch(self.avg(x).squeeze()))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        # route_prob_max, idx = torch.max(route_prob, dim=-1, keepdims=True)

        # # Get indexes of tokens going to each expert
        # indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(2)]
        
        idx = torch.argmax(route_prob, dim=-1, keepdims=True)
        mask = torch.zeros_like(idx).scatter_(0, idx, 1.).view(-1, 1, 1, 1)
        mask2 = 1 - mask
        # print(mask.shape, x1.shape, x2.shape)
        x = x1 * mask + x2 * mask2
        # x = torch.cat((x1,x2), 1)        
   
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class MixConvDy(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1a = DYModule(in_features, in_features, fc_squeeze=fc_squeeze)
        self.fc1b = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features+in_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.fc1a(x)
        x2 = self.fc1b(x)
        x1 = self.act(x1)
        x2 = self.act(x2)
        x = torch.cat((x1,x2), 1)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyWeightGroupConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DyWeightGroupConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)        
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1, 1),
            nn.AdaptiveAvgPool2d(kernel_size),
        )
        self.groups = groups
        self.out_channels = out_channels
        self.act = Hsigmoid()

    def forward(self, x):
        bs, Ci, H, W = x.shape
        weight = self.weight
        ws = weight.shape
        weight_ = self.trans(x)
        weight_ = self.act(weight_).unsqueeze(1)
        x = x.view(1, -1, H, W)
        # print(weight.shape, weight_.shape)
        w = weight.unsqueeze(0) * weight_
        # print(w.shape)
        w = w.view(bs*ws[0], ws[1], ws[2], ws[3])

        b = self.bias.unsqueeze(0) * weight_.view(bs, 1, -1).mean(2)
        b = b.view(-1)
       
        conv = nn.functional.conv2d(x, w, b, self.stride, 
                              self.padding, self.dilation, self.groups*bs)
        _, _, h, w = conv.shape
        conv=conv.reshape(bs,ws[0], h, w)
        return conv


class DyWeightConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DyWeightConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)        
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1, 1),
            nn.AdaptiveAvgPool2d(kernel_size),
        )
        self.act = Hsigmoid()

    def forward(self, x):
        weight = self.weight
        weight_ = self.trans(x).mean(0)
        weight_ = self.act(weight_).unsqueeze(0)
        # print(weight.shape, weight_.shape)
        w = weight * weight_
       
        conv = nn.functional.conv2d(x, w, self.bias, self.stride, 
                              self.padding, self.dilation, self.groups)
        return conv


class AttnConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(AttnConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)        
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.squeeze_dim = 8
        self.scale = self.squeeze_dim ** -0.5
        attn_drop = 0.01
        self.trans = nn.Sequential(
            # nn.Conv2d(in_channels, self.squeeze_dim, 1, 1),
            nn.AdaptiveAvgPool2d(kernel_size),
        )
        self.qk = nn.Linear(in_channels, self.squeeze_dim * 2, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        xorg = x
        x = self.trans(x)
        shape = x.shape
        if len(shape) == 4:
            B, C, H, W = shape
            N = H * W
            x = torch.flatten(x, start_dim=2).transpose(-2, -1) # (B, N, C)
        weight = self.weight
        
        qk = self.qk(x).reshape(B, N, 2, 1, self.squeeze_dim).permute(2, 0, 3, 1, 4)
        q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        weight_ = attn.mean(0).mean(-2).unsqueeze(0)

        w = weight * weight_.reshape(1, 1, self.kernel_size, self.kernel_size)
       
        conv = nn.functional.conv2d(xorg, w, self.bias, self.stride, 
                              self.padding, self.dilation, self.groups)
        return conv



class DyMlpBigEff(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = DYModule(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.act = act_layer()
        self.fc2 = DYModule(hidden_features, out_features, fc_squeeze=fc_squeeze)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyMlpEff(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = DYModuleSmall(in_features, hidden_features, fc_squeeze=fc_squeeze)
        self.act = act_layer()
        self.fc2 = DYModuleSmall(hidden_features, out_features, fc_squeeze=fc_squeeze)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DyMlpSplit(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SplitDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DyMlpUnequalSplit(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = UnequalSplitDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DyMlpV2(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = DYModule(in_features, hidden_features)
        self.fc1 = DynamicConv(in_features, in_features // 2, 1, 1)
   
        self.act = act_layer()
        self.fc2a = nn.Conv2d(in_features // 2, hidden_features, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.se = SELayer(out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2a(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.se(x)
        x = self.drop(x)
        return x



class DyMlp2(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1a = DYModule(in_features, hidden_features // 2)
        self.fc1b = DYModule(in_features, hidden_features // 4)
       
        self.act = act_layer()
        self.fc2a = nn.Conv2d(hidden_features // 2, out_features // 2, 1, groups=1)
        self.fc2b = nn.Conv2d(hidden_features // 4, out_features // 2, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.fc1a(x)
        x1 = self.act(x1)
        x1 = self.drop(x1)
        x2 = self.fc1b(x)
        x2 = self.act(x2)
        x2 = self.drop(x2)
        # x = torch.cat([x1, x2], 1)
        x1 = self.fc2a(x1)
        x1 = self.act(x1)
        x1 = self.drop(x1)
        x2 = self.fc2b(x2)
        x2 = self.act(x2)
        x2 = self.drop(x2)
        x = torch.cat([x1, x2], 1)
        return x


class DyMlp3(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LightDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyMlpShared(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LightDynamicConvV3(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyMlp4(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LightDynamicConvV4(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyMlpGCT(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LightDynamicConvGCT(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyMlpDecompose(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = DecomposeDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyMlpSubspaceDecompose(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SharedSubSpaceDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DyMlpSubspaceDecomposeV1(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SharedSubSpaceDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


    @staticmethod
    def w(t,i,p,t_array):
        w=(t-t_array[i])/(t_array[i+p]-t_array[i])
        return w
    
    @staticmethod
    def B(l,t,i,p,t_array):

        if t[l]>=t_array[i] and t[l]<=t_array[i+1] and p==0:
            k=1
        elif p>0:
            k=w(t[l],i,p,t_array)*B(l,t,i,p-1,t_array)+(1-w(t[l],i+1,p,t_array))*B(l,t,i+1,p-1,t_array)
        else:
            k=0
        return k


class DyMlpConvShared(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.dy = DySubspace(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        inp = x
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x, soft_att = self.dy(x, inp)
        x = self.act(x)
        x = self.fc2(x)
        # x += inp
        bs,in_planels,h,w=x.shape
        K = soft_att.shape[-1]
        xs = [x, inp]
        x = torch.stack(xs, 1)
        x = x.view(bs,K,-1)
        x = torch.matmul(soft_att, x).view(bs, in_planels, h, w) #+ x
        # x = self.drop(x)
        return x



class DyMlpConvBNRes(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SeparableDecomposeDynamicConvBNRes(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ODConvMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=1/16):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ODConv2d(in_features, hidden_features, 1, 1, reduction=fc_squeeze)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DyMlpConvBNFull(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SeparableDecomposeDynamicConvBNFull(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class SwitchDyConvBNMlp(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dyconv1 = SeparableDecomposeDynamicConvBNFull(in_features, hidden_features, 1, 1)
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1)
        self.fc1 = SwitchDyConvBN(expert1=self.dyconv1, expert2=self.conv1, d_model=in_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DyMlpConvBNV1(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SeparableDecomposeDynamicConvBNV1(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DyMlpConvBN(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SeparableDecomposeDynamicConvBN(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyMlpReuse(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ReuseDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DyMlpMulti(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = MultiDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyMlpSeparableDecomposeGumbel(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SeparableDecomposeGumbelDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class AdapterFC2(nn.Module):
    expansion = 1

    def __init__(self, dim, in_dim, out_dim, groups=1,bias=True,K=4,squeeze=8,temprature=30,ratio=4,init_weight=True, act_layer=nn.GELU, pca=True):
        super(AdapterFC2, self).__init__()
        self.pca = pca
        # self.bn1 = nn.BatchNorm2d(dim)
        self.conv_wh = nn.Conv2d(dim, in_dim, 1)
        self.K=K
        self.out_dim = out_dim
        self.in_dim = in_dim

        # self.attention=Attention(in_planes=dim,ratio=ratio,K=K,temprature=temprature,init_weight=init_weight)
        #for i in self.conv_wh.parameters():
        #    i.requires_grad = False
        # if pca==True:
        #     self.conv_MAL = nn.Conv2d(in_dim, K*in_dim, 1, 1, 0, bias=bias, groups=1)
        self.conv_rc = nn.Conv2d(in_dim, out_dim, 1)
        #for i in self.conv_rc.parameters():
        #    i.requires_grad = False        
        # self.bn2 = nn.BatchNorm2d(out_dim)
        self.act = act_layer()


    def forward(self, x):
        bs,in_planes,h,w=x.shape
        # softmax_att=self.attention(x).view(bs, 1, self.K)
        # residual = x

        mean_wh_ = torch.mean(x.view(bs, in_planes, -1), axis=-1, keepdim=True)
        x_ = x.view(bs, in_planes, -1) - mean_wh_
        x = x_.view(bs, in_planes, h, w)
        
        # out = self.bn1(x)
        out = self.conv_wh(x)#[:, :self.in_dim, :, :]
        # # if self.pca == True:
        # #     out = self.conv_MAL(out)
        # #     out = out.view(bs,self.K,-1)
        # #     out = torch.matmul(softmax_att, out).view(bs,self.in_dim,h,w)    
        # mean_rc_ = torch.mean(out.view(bs, self.in_dim, -1), axis=-1, keepdim=True)
        # out_ = out.view(bs, self.in_dim, -1) + mean_rc_
        # out = out_.view(bs, self.in_dim, h, w)
        out = self.conv_rc(out)

        # mean_rc_ = torch.mean(out.view(bs, self.out_dim, -1), axis=-1, keepdim=True)
        # out_ = out.view(bs, self.out_dim, -1) + mean_rc_
        # out = out_.view(bs, self.out_dim, h, w)
        
        # out += residual
        # out = self.bn2(out)  
        # out = self.act(out)          

        return out


class PCADyMlpSeparableDecompose(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None, name='',
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8, pca=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.use_pca = pca
        self.fc1 = SeparableDecomposeDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = AdapterFC2(hidden_features, out_features // 1, out_features, pca=pca)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # if not self.use_pca:
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyMlpSeparableDecompose(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8, pca=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.use_pca = pca
        self.fc1 = SeparableDecomposeDynamicConv(in_features, hidden_features, 1, 1, pca=pca)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        if not self.use_pca:
            x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DyMlpSeparableDecomposeLast(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = SeparableDecomposeDynamicConv(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyMlpSeparableDecomposeBoth(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SeparableDecomposeDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = SeparableDecomposeDynamicConv(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class LUDyMlpDecompose(nn.Module):
    """
    Implementation of Dynamic MLP with one dynamic convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., fc_squeeze=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LUDecomposeDynamicConv(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SVDMlpFinetune(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='network',
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        svd = pickle.load(open('/home/deepak/poolformer/output/train/20220927-142718-fewseparabledecomposedyconvformer_ppaa_s12_224-224/' + '{}.fc2.weight'.format(name) + '_svd.sav', 'rb'))
        u = torch.tensor(svd['u'], requires_grad=False).cuda()
        v = torch.tensor(svd['v'], requires_grad=False).cuda()

        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.s=nn.Parameter(torch.randn(in_features),requires_grad=True).cuda()
        self.S_=torch.diag(self.s).cuda()
        self.S_=torch.cat((self.S_, torch.zeros(in_features, hidden_features - in_features).cuda()), 1).cuda()
        self.weight = torch.mm(torch.mm(u, self.S_), v.t()).unsqueeze(-1).unsqueeze(-1).cuda()
        self.bias=nn.Parameter(torch.randn(out_features),requires_grad=True).cuda()
        self.act = act_layer()
        self.fc1 = SeparableDecomposeDynamicConv(in_features, hidden_features, 1, 1)
        self.drop = nn.Dropout(drop)

        self.stride=1
        self.padding=0
        self.dilation=1
        self.groups=1

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = torch.nn.functional.conv2d(x,weight=self.weight,bias=self.bias,stride=self.stride,padding=self.padding,groups=self.groups,dilation=self.dilation)
        x = self.drop(x)
        return x





class SVDMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='',
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        svd = pickle.load(open('/home/deepak/poolformer/output/train/20220927-142718-fewseparabledecomposedyconvformer_ppaa_s12_224-224/' + '{}.fc2.weight'.format(name) + '_svd.sav', 'rb'))
        self.u = torch.tensor(svd['u'], requires_grad=False).cuda()
        self.v = torch.tensor(svd['v'], requires_grad=False).cuda()
        self.zeros_ = torch.zeros(in_features, hidden_features - in_features, requires_grad=False, device='cuda')
        self.register_buffer('u', self.u)
        self.register_buffer('v', self.v)
        self.register_buffer('zeros_', self.zeros_)

        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.s=nn.Parameter(torch.randn(in_features, device='cuda'))
        
        self.bias=nn.Parameter(torch.randn(out_features, device='cuda'))
        self.act = act_layer()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.drop = nn.Dropout(drop)

        self.stride=1
        self.padding=0
        self.dilation=1
        self.groups=1

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        S_=torch.diag(self.s)
        S_=torch.cat((S_, self.zeros_), 1)
        self.weight = torch.mm(torch.mm(self.u, S_), self.v.t()).unsqueeze(-1).unsqueeze(-1)
        x = torch.nn.functional.conv2d(x,weight=self.weight,bias=self.bias,stride=self.stride,padding=self.padding,groups=self.groups,dilation=self.dilation)
        x = self.drop(x)
        return x



class UnrollDyMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, n = 3,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        

        self.n = n
        self.fc_module = nn.ModuleList()
        self.attn_module = nn.ModuleList()
        dims = []
        for k in range(self.n):
            if k == 0:
                in_dim = in_features
                fc1 = DYModule(in_dim, in_features, fc_squeeze=8)
            else:
                in_dim = dims[k - 1] + in_features
                fc1 = nn.Conv2d(in_dim, in_features, 1)
            attention=GCTAttention(in_planes=in_features, groups=1)
            self.fc_module.append(fc1)
            self.attn_module.append(attention)
            dims.append(in_dim)
        self.act = act_layer()
       
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for k in range(self.n):
            input = x
            y = self.fc_module[k](x)
            attn = self.attn_module[k](y)
            x = y * attn
            if k != self.n - 1:
                x = self.act(x)
            x = self.drop(x)
            if k != self.n - 1:
                x = torch.cat((input, x), 1)
       
        return x


class UnrollMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, n = 3,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.n = n
        self.fc_module = nn.ModuleList()
        dims = []
        for k in range(self.n):
            if k == 0:
                in_dim = in_features
                fc1 = nn.Conv2d(in_dim, in_features, 1)
            else:
                in_dim = dims[k - 1] + in_features
                fc1 = nn.Conv2d(in_dim, in_features, 1)
            self.fc_module.append(fc1)
            dims.append(in_dim)
        self.act = act_layer()
       
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for k in range(self.n):
            input = x
            x = self.fc_module[k](x)
            if k != self.n - 1:
                x = self.act(x)
            x = self.drop(x)
            if k != self.n - 1:
                x = torch.cat((input, x), 1)
       
        return x



class DynamicMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='',
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ODynamicConv(in_features, out_features, 1, 1, K=8)
        # self.fc1 = DYModule(in_features, out_features, fc_squeeze=4)
        self.act = act_layer()
        
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.act(x)
        y = x
        # x = self.drop(x)
        return x



class DynamicChannelMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='',
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = DynamicChannelConv(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        y = x
        x = self.drop(x)
        return x


class GCTMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='',
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.gct1 = GCTAttention(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.gct2 = GCTAttention(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = x*self.gct1(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = x*self.gct2(x)
        x = self.drop(x)
        return x



class CascadeMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='',
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, 1)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, out_features, 1)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', use_cca=False,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x


    
class GroupConvMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=2,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = hidden_features ** -0.5
        # self.weight_soft = nn.Parameter(torch.randn(2, 1, out_features, 1, 1))
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=1)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.branch = nn.Conv2d(in_features, in_features, 1, groups=in_features)
       
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(2 * in_features, in_features, 1, groups=in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # weight_soft = self.softmax(self.weight_soft)

        sh = x.shape
        attn = self.branch(x)
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        x = torch.cat((x, attn), 1)
        x = self.proj(x)
        # x = weight_soft[0] * x + weight_soft[1] * attn
        return x



class GroupLiteGCTAttMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = in_features ** -0.5
        self.weight_soft = GCTAttention(in_features)
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.k = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        # self.proj = nn.Conv2d(2 * in_features, in_features, 1, groups=in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        weight_soft = self.weight_soft(x)

        sh = x.shape
        # trick here to make q@k.t more stable
        attn = (x.view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        # x = torch.cat((x, attn), 1)
        # x = torch.cat((weight_soft[0] * x, weight_soft[1] * attn), 1)
        # x = self.proj(x)
        x = weight_soft * x + (1 - weight_soft) * attn
        return x


class GroupGCTAttMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = in_features ** -0.5
        self.weight_soft = GCTAttention(in_features)
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=1)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.k = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        # self.proj = nn.Conv2d(2 * in_features, in_features, 1, groups=in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        weight_soft = self.weight_soft(x)

        sh = x.shape
        # trick here to make q@k.t more stable
        attn = (x.view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        # x = torch.cat((x, attn), 1)
        # x = torch.cat((weight_soft[0] * x, weight_soft[1] * attn), 1)
        # x = self.proj(x)
        x = weight_soft * x + (1 - weight_soft) * attn
        return x



class MicroBlockMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = hidden_features ** -0.5
        # self.weight_soft = nn.Parameter(torch.randn(2, 1, out_features, 1, 1))
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        
        t1 = (1, 1)
        gs1 = (0, 4)
        gs2 = (0, 8, 8)
        activation_cfg = {
            'MODULE' : 'DYShiftMax',
            'ACT_MAX' : 2.0,
            'LINEARSE_BIAS' : False,
            'REDUCTION' : 8,
            'INIT_A' : [1.0, 1.0],
            'INIT_B' : [0.0, 0.0],
            'INIT_A_BLOCK3' : [1.0, 0.0],
            'dy' : [2, 0, 1],
            'ratio' : 1,    
        }

        self.fc1 = DYMicroBlock(in_features, hidden_features,
            kernel_size=1, 
            stride=1, 
            ch_exp=t1, 
            ch_per_group=gs1, 
            groups_1x1=gs2,
            depthsep = True,
            shuffle = True,
            pointwise = 'group',
            activation_cfg=activation_cfg,
        )
        self.act = act_layer()
        self.fc2 = DYMicroBlock(hidden_features, out_features,
            kernel_size=1, 
            stride=1, 
            ch_exp=t1, 
            ch_per_group=gs1, 
            groups_1x1=gs2,
            depthsep = True,
            shuffle = True,
            pointwise = 'group',
            activation_cfg=activation_cfg,
        )
        
       
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        # self.proj = nn.Conv2d(2 * in_features, in_features, 1, groups=in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        x = self.fc1(x)
        # x = self.bn1(x)
        # x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        
        return x



class GroupAttMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=2,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = hidden_features ** -0.5
        self.weight_soft = nn.Parameter(torch.randn(2, 1, out_features, 1, 1))
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.k = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        # self.proj = nn.Conv2d(2 * in_features, in_features, 1, groups=in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        weight_soft = self.softmax(self.weight_soft)

        sh = x.shape
        # trick here to make q@k.t more stable
        attn = (x.view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        # x = torch.cat((x, attn), 1)
        # x = torch.cat((weight_soft[0] * x, weight_soft[1] * attn), 1)
        # x = self.proj(x)
        
        x = weight_soft[0] * x + weight_soft[1] * attn
        return x







class GroupLiteAttMlpBN(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = hidden_features ** -0.5

        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.bn2 = nn.BatchNorm2d(out_features)
        
        self.weight_soft = nn.Parameter(torch.randn(2))
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
    
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.k = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
    
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        weight_soft = self.softmax(self.weight_soft)

        sh = x.shape
        # trick here to make q@k.t more stable
        attn = (x.view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        x = self.bn2(x)
        y = x
        x = self.drop(x)
        x = weight_soft[0] * x + weight_soft[1] * attn
        return x



class GroupLiteAttMlpV2(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = in_features ** -0.5
        
        self.weight_soft = nn.Parameter(torch.randn(2))
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
    
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.k = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
    
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        weight_soft = self.softmax(self.weight_soft)

        sh = x.shape
        # trick here to make q@k.t more stable
        attn = (x.view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        x = weight_soft[0] * x + weight_soft[1] * attn
        return x


class GroupLiteAttMlpV4(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = in_features ** -0.5
        print(in_features, hidden_features)
        
        # self.weight_soft = nn.Parameter(torch.randn(2))
        self.weight_soft = Attention(in_features,K=2, ratio=4)
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
    
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.aux = nn.Conv2d(in_features, in_features, 1, groups=1)
        self.v = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
    
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        weight_soft = self.weight_soft(x)

        attn = self.aux(x)
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        x = weight_soft[:, 0].view(-1, 1, 1, 1) * x + weight_soft[:, 1].view(-1, 1, 1, 1) * attn
        return x



class GroupLiteAttMlpV3Old(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = in_features ** -0.5
        
        self.weight_soft = Attention(in_features,K=2, ratio=4)
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
    
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.k = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
    
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        weight_soft = self.weight_soft(x)

        sh = x.shape
        # trick here to make q@k.t more stable
        attn = (x.view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        x = weight_soft[:, 0].view(-1, 1, 1, 1) * x + weight_soft[:, 1].view(-1, 1, 1, 1) * attn
        return x


class GroupLiteAttMlpV3(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = in_features ** -0.5
        
        self.weight_soft = Attention(in_features,K=2, ratio=4)
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
    
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.k = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Conv2d(in_features, in_features, 1, groups=in_features)
    
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        sh = x.shape
        # trick here to make q@k.t more stable
        attn = (x.view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        weight_soft = self.weight_soft(x)
        y = x
        x = self.drop(x)
        x = weight_soft[:, 0].view(-1, 1, 1, 1) * x + weight_soft[:, 1].view(-1, 1, 1, 1) * attn
        return x



class GroupLiteAttMlpV3A(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = in_features ** -0.5
        
        self.weight_soft = Attention(in_features,K=2, ratio=4)
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
    
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.q = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.k = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Conv2d(in_features, in_features, 1, groups=in_features)
    
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        sh = x.shape
        # trick here to make q@k.t more stable
        attn = (self.q(x).view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        weight_soft = self.weight_soft(x)
        y = x
        x = weight_soft[:, 0].view(-1, 1, 1, 1) * x + weight_soft[:, 1].view(-1, 1, 1, 1) * attn
        x = self.drop(x)
       
        return x


class GroupAttConCatMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=2,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        self.scale = in_features ** -0.5
        # self.weight_soft = nn.Parameter(torch.randn(2, 1, out_features, 1, 1))
        self.softmax = nn.Softmax(0)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.k = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(2 * in_features, in_features, 1, groups=in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # weight_soft = self.softmax(self.weight_soft)

        sh = x.shape
        # trick here to make q@k.t more stable
        attn = (x.view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        x = torch.cat((x, attn), 1)
        # x = torch.cat((weight_soft[0] * x, weight_soft[1] * attn), 1)
        x = self.proj(x)
        # x = weight_soft[0] * x + weight_soft[1] * attn
        return x


class GroupMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=1)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x


class GaussConvMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=1,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = GaussConv(in_features, hidden_features, 1, groups=1)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = GaussConv(hidden_features, out_features, 1, groups=groups)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x


class GaussConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, var=1,
                 groups=1, bias=True):
        super(GaussConv, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=groups
        self.bias=bias
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        # self.var = torch.tensor(var)
  
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.mean = nn.Parameter(torch.randn(1, 1, kernel_size, kernel_size))
        self.var = nn.Parameter(torch.randn(1, 1, kernel_size, kernel_size))
        if(bias):
            self.bias = nn.Parameter(torch.randn(out_channels),requires_grad=True)
            self.bias_mean = nn.Parameter(torch.randn(out_channels),requires_grad=True)
            self.bias_var = nn.Parameter(torch.randn(out_channels),requires_grad=True)
        else:
            self.bias=None


    def forward(self, x):
        # x_h, x_l = x if type(x) is tuple else (x, None)
        weight = torch.exp(-(self.weight-self.mean)**2/(2*self.var**2)) / torch.sqrt(2*torch.pi*self.var**2)
        
        if(self.bias is not None):
            bias = torch.exp(-(self.bias-self.bias_mean)**2/(2*self.bias_var**2)) / torch.sqrt(2*torch.pi*self.bias_var**2)
            output=F.conv2d(x,weight=weight,bias=bias,stride=self.stride,padding=self.padding,groups=self.groups,dilation=self.dilation)
        else:
            output=F.conv2d(x,weight=weight,bias=None,stride=self.stride,padding=self.padding,groups=self.groups,dilation=self.dilation)
        
        # output=output.reshape(bs,self.out_planes,h,w)
        return output

        


class GroupSortMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        self.in_channels_per_group = in_features // groups
        self.hidden_channels_per_group = hidden_features // groups
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def batched_index_select(input, dim, index):
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


    def forward(self, x):
        x, indices1 = torch.sort(x, 1)
        # route_prob_max, routes = torch.topk(x.view(-1, x.size(1)), self.in_features, 1)
        # y1 = self.batched_index_select(x, 1, routes[:, :self.in_channels_per_group])
        # y2 = self.batched_index_select(x, 1, routes[:, self.hidden_features//self.ratio-1:])

        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x, indices2 = torch.sort(x, 1)
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x





class GroupAvgConnectMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        self.groups = groups
        self.in_channels_per_group = in_features // groups
        self.hidden_channels_per_group = hidden_features // groups
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.weights1 = nn.Parameter(torch.randn(groups))
        self.weights2 = nn.Parameter(torch.randn(groups))
        
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        self.group_c1 = nn.Conv2d(groups, groups, 1)
        self.group_c2 = nn.Conv2d(groups, groups, 1)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def batched_index_select(input, dim, index):
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        gc1, indices = group_avg_interconnect(x, self.groups)
        gc1 = self.group_c1(gc1)
        for ii, index in enumerate(indices):
            x[:, ii*self.hidden_channels_per_group:(ii+1)*self.hidden_channels_per_group] += self.weights1[ii].sigmoid() * gc1[:, ii].unsqueeze(1)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x = self.fc2(x)
        gc2, indices2 = group_avg_interconnect(x, self.groups)
        gc2 = self.group_c2(gc2)
        for ii, index in enumerate(indices2):
            x[:, ii*self.in_channels_per_group:(ii+1)*self.in_channels_per_group] += self.weights2[ii].sigmoid() * gc2[:, ii].unsqueeze(1)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x


class GroupMixConnectAttnMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        self.groups = groups
        self.in_channels_per_group = in_features // groups
        self.hidden_channels_per_group = hidden_features // groups
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.scale = hidden_features ** -0.5

        self.weight_soft = nn.Parameter(torch.randn(2))
        self.weight_connect1 = nn.Parameter(torch.randn(hidden_features))
        self.weight_connect2 = nn.Parameter(torch.randn(out_features))
        self.softmax = nn.Softmax(0)

        # self.weights1 = nn.Parameter(torch.randn(groups))
        # self.weights2 = nn.Parameter(torch.randn(groups))
        
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        # self.group_c1 = nn.Conv2d(groups, groups, 1)
        # self.group_c2 = nn.Conv2d(groups, groups, 1)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.k = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
    
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

        self.index1 = self.index_perm(hidden_features, self.groups, self.hidden_channels_per_group)
        self.index2 = self.index_perm(out_features, self.groups, self.in_channels_per_group)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def index_perm(self, inp, g, gc):
        index=torch.Tensor(range(inp))
        # print(inp, g, gc)
        if g <= gc:
            for u, p in enumerate(reversed(range(g))):
                for q in range(p):
                    # print(q + gc*u + u, (u+q+1)*gc + u)
                    index[[q + gc*u + u, (u+q+1)*gc + u]] = index[[(u+q+1)*gc + u, q + gc*u + u]]
        else:
            for ggg in range(g // gc):
                for u, p in enumerate(reversed(range(gc))):
                    for q in range(p):
                        index[[gc*ggg + q + gc*u + u, gc*ggg + (u+q+1)*gc + u]] = index[[gc*ggg + (u+q+1)*gc + u, gc*ggg + q + gc*u + u]]

        return index.view(inp).type(torch.LongTensor)

    def forward(self, x):

        weight_soft = self.softmax(self.weight_soft)

        sh = x.shape
        # trick here to make q@k.t more stable
        attn = (x.view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        

        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x2 = x[:,self.index1,:,:]
        x += x2 * self.weight_connect1.sigmoid().view(1, -1, 1, 1)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x = self.fc2(x)
        x3 = x[:,self.index2,:,:]
        x += x3 * self.weight_connect2.sigmoid().view(1, -1, 1, 1)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        x = weight_soft[0] * x + weight_soft[1] * attn
        return x



class GroupAvgConnectAttnMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        self.groups = groups
        self.in_channels_per_group = in_features // groups
        self.hidden_channels_per_group = hidden_features // groups
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.scale = hidden_features ** -0.5

        self.weight_soft = nn.Parameter(torch.randn(2))
        self.softmax = nn.Softmax(0)

        self.weights1 = nn.Parameter(torch.randn(groups))
        self.weights2 = nn.Parameter(torch.randn(groups))
        
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        self.group_c1 = nn.Conv2d(groups, groups, 1)
        self.group_c2 = nn.Conv2d(groups, groups, 1)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.k = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
    
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def batched_index_select(input, dim, index):
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


    def forward(self, x):

        weight_soft = self.softmax(self.weight_soft)

        sh = x.shape
        # trick here to make q@k.t more stable
        attn = (x.view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        

        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        gc1, indices = group_avg_interconnect(x, self.groups)
        gc1 = self.group_c1(gc1)
        feats1 = []
        for ii, index in enumerate(indices):
            feats1.append(self.weights1[ii].sigmoid() * gc1[:, ii].unsqueeze(1))
        feats1 = torch.cat(feats1, 1)
        feats1 = torch.repeat_interleave(feats1, self.hidden_channels_per_group, dim=1)
        x += feats1
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x = self.fc2(x)
        gc2, indices2 = group_avg_interconnect(x, self.groups)
        gc2 = self.group_c2(gc2)
        feats2 = []
        for ii, index in enumerate(indices2):
            feats2.append(self.weights2[ii].sigmoid() * gc2[:, ii].unsqueeze(1))
        feats2 = torch.cat(feats2, 1)
        feats2 = torch.repeat_interleave(feats2, self.in_channels_per_group, dim=1)
        x += feats2
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        x = weight_soft[0] * x + weight_soft[1] * attn
        return x

class GroupConnectMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        self.groups = groups
        self.in_channels_per_group = in_features // groups
        self.hidden_channels_per_group = hidden_features // groups
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features


        
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        self.group_c1 = nn.Conv2d(groups, groups, 1)
        self.group_c2 = nn.Conv2d(groups, groups, 1)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def batched_index_select(input, dim, index):
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        gc1, indices = group_interconnect(x, self.groups)
        gc1 = self.group_c1(gc1)
        for ii, index in enumerate(indices):
            x[:, index] = gc1[:, ii].unsqueeze(1)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x = self.fc2(x)
        gc2, indices2 = group_interconnect(x, self.groups)
        gc2 = self.group_c2(gc2)
        for ii, index in enumerate(indices2):
            x[:, index] = gc2[:, ii].unsqueeze(1)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x



class GroupConnectMaxMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        self.groups = groups
        self.in_channels_per_group = in_features // groups
        self.hidden_channels_per_group = hidden_features // groups
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        self.group_c1 = nn.Conv2d(groups, groups, 1)
        self.group_c2 = nn.Conv2d(groups, groups, 1)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def batched_index_select(input, dim, index):
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


    def forward(self, x):
        bs = x.shape[0]
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        gc1, indices = group_max_interconnect(x, self.groups)
        print(gc1.shape)
        gc1 = self.group_c1(gc1)
        for ii, index in enumerate(indices):
            x[:, index] = gc1[ii].unsqueeze(1) #np.arange(bs), 
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x = self.fc2(x)
        gc2, indices2 = group_max_interconnect(x, self.groups)
        gc2 = self.group_c2(gc2)
        for ii, index in enumerate(indices2):
            x[:, index] = gc2[np.arange(bs), ii].unsqueeze(1)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x



class GroupConnectMaxAttnMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        self.groups = groups
        self.in_channels_per_group = in_features // groups
        self.hidden_channels_per_group = hidden_features // groups
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.scale = in_features ** -0.5
        
        self.weight_soft = Attention(in_features,K=2, ratio=4)
        self.softmax = nn.Softmax(0)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        self.group_c1 = nn.Conv2d(groups, groups, 1)
        self.group_c2 = nn.Conv2d(groups, groups, 1)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.k = nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Conv2d(in_features, in_features, 1, groups=in_features)
    
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def batched_index_select(input, dim, index):
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


    def forward(self, x):
        sh = x.shape
        bs = sh[0]
        # trick here to make q@k.t more stable
        attn = (x.view(x.size(0), x.size(1), -1) * self.scale) @ self.k(x).reshape(x.size(0), -1, x.size(1))
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn = (attn @ self.v(x).view(x.size(0), x.size(1), -1)).view(sh[0], sh[1], sh[2], sh[3])
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        gc1, indices = group_max_interconnect(x, self.groups)
    
        gc1 = self.group_c1(gc1)
        indices = torch.cat(indices, 1)
        x = x.clone().scatter_(1, indices, gc1)
        # for ii, index in enumerate(indices):
        #     # print(x[np.arange(bs), index].shape)
        #     x[index[:,0, 0, 0]] = gc1[:, ii].unsqueeze(1) #np.arange(bs), 
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x = self.fc2(x)
        gc2, indices2 = group_max_interconnect(x, self.groups)
        gc2 = self.group_c2(gc2)
        indices2 = torch.cat(indices2, 1)
        x = x.clone().scatter_(1, indices2, gc2)
        # for ii, index in enumerate(indices2):
            # x[:, index] = gc2[np.arange(bs), ii].unsqueeze(1)
        # x = self.bn2(x)
        y = x
        weight_soft = self.weight_soft(x)
        x = weight_soft[:, 0].view(-1, 1, 1, 1) * x + weight_soft[:, 1].view(-1, 1, 1, 1) * attn
        x = self.drop(x)
        
        return x



class GroupMlpV2(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=2,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x


class HighMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='', groups=4, K=2,
                 out_features=None, act_layer=nn.GELU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        self.K = K
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.weight_soft = nn.Parameter(torch.randn(K+1))
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=1)
        self.softmax = nn.Softmax(0)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        weight_soft = self.softmax(self.weight_soft)
        input = x
        x = x * self.weight_soft[-1]
        for k in range(self.K):
            x = x + weight_soft[k] * input**(k+2) 
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x


class SlidingMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='',
                 out_features=None, act_layer=nn.GELU, drop=0., ratio=2):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=1)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        # self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=4)
        self.fc2 = nn.Conv3d(1, out_features, (hidden_features//ratio, 1, 1), padding=(0, 0, 0))
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = x.unsqueeze(1)
        x = self.fc2(x)
        # print(x.shape)
        x = x.mean(2)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x


class GroupReparamMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='',
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = GroupReparam_1x1(in_features, hidden_features, 1)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = GroupReparam_1x1(hidden_features, out_features, 1)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x


class GroupRepVGGMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='',
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = GroupRepBlock(hidden_features, out_features, 1, groups=4, deploy=True)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x


class DyReparamMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, name='',
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = DyReparam_1x1(in_features, hidden_features, 1)
        # self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = DyReparam_1x1(hidden_features, out_features, 1)
        # self.bn2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop(x)
        return x


