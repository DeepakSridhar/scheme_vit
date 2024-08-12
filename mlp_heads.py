import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers.helpers import to_2tuple


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

class Attention(nn.Module):
    def __init__(self,in_planes,ratio,K,temprature=30,init_weight=True):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.temprature=temprature
        assert in_planes>ratio
        hidden_planes=in_planes//ratio
        self.net=nn.Sequential(
            nn.Conv2d(in_planes,hidden_planes,kernel_size=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes,K,kernel_size=1,bias=False)
        )

        if(init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if(self.temprature>1):
            self.temprature-=1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        att=self.avgpool(x) #bs,dim,1,1
        att=self.net(att).view(x.shape[0],-1) #bs,K
        return F.softmax(att/self.temprature,-1)


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) == 4:
            b, c, _, _ = x.size()
            y = self.avg_pool2d(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
        else:
            b, c, _ = x.size()
            y = self.avg_pool1d(x).view(b, c)
            y = self.fc(y).view(b, c, 1)        
        
        return x * y.expand_as(x)


def group_interconnect(x, groups=4):
    channels_per_group = x.shape[1] // groups
    # chunks = torch.split(x, groups)
    feats = []
    indices = []
    for i in range(groups):
        rint = torch.randint(i*channels_per_group, i*channels_per_group + channels_per_group, (1,))
        feat = x[:, rint]
        feats.append(feat)
        indices.append(rint)
    out = torch.stack(feats, 1).squeeze(2)
    return out, indices


def group_avg_interconnect(x, groups=4):
    channels_per_group = x.shape[1] // groups
    # chunks = torch.split(x, groups)
    feats = []
    indices = []
    for i in range(groups):
        # rint = torch.randint(i*channels_per_group, i*channels_per_group + channels_per_group, (1,))
        feat = x[:, i*channels_per_group:(i+1)*channels_per_group].mean(1)
        feats.append(feat)
        indices.append(np.arange(i*channels_per_group,(i+1)*channels_per_group))
    out = torch.stack(feats, 1).squeeze(2)
    return out, indices

def group_max_interconnect(x, groups=4):
    channels_per_group = x.shape[1] // groups
    chunks = torch.split(x, groups, 1)
    # print(chunks[0].shape)
    bs = x.shape[0]
    feats = []
    indices = []
    for i in range(groups):
        rint = torch.argmax(chunks[i], 1, keepdim=True)
        # _, rint = torch.max(chunks[i], dim=1)
        # print(rint.shape)
        rint += i*channels_per_group 
        feat = torch.gather(x, 1, rint)
        # feat = x[np.arange(bs), rint]
        feats.append(feat)
        indices.append(rint)
    out = torch.stack(feats, 1).squeeze(2)
    return out, indices


class GroupLiteAttMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, mlp_ratio=4, name='', groups=4,
                 out_features=None, act_layer=StarReLU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)
        self.scale = hidden_features ** -0.5
        
        self.weight_soft = nn.Parameter(torch.randn(2))
        self.softmax = nn.Softmax(0)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
    
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
        
        self.k = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.v = nn.Identity() #nn.Conv2d(in_features, in_features, 1, groups=in_features)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        weight_soft = self.softmax(self.weight_soft)
        x = x.permute(0, 3, 1, 2)
        if self.training:
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
        x = self.drop1(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        y = x
        x = self.drop2(x)
        if self.training:
            x = weight_soft[0] * x + weight_soft[1] * attn

        x = x.permute(0, 2, 3, 1)
        return x


class PartialGroupLiteAttMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, mlp_ratio=4, name='', groups=2,
                 out_features=None, act_layer=StarReLU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        self.scale = hidden_features ** -0.5
        
        self.weight_soft = nn.Parameter(torch.randn(2))
        self.softmax = nn.Softmax(0)
        
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=1)
    
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
        x = x.permute(0, 3, 1, 2)
        if self.training or True:
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
        if self.training or True:
            x = weight_soft[0] * x + weight_soft[1] * attn
        x = x.permute(0, 2, 3, 1)
        return x


class GroupConnectAttnMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, mlp_ratio=4, name='', groups=4,
                 out_features=None, act_layer=StarReLU, drop=0., shift=False):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        self.groups = groups
        self.in_channels_per_group = in_features // groups
        self.hidden_channels_per_group = hidden_features // groups
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
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
        gc1, indices = group_interconnect(x, self.groups)
        gc1 = self.group_c1(gc1)
        for ii, index in enumerate(indices):
            x[:, index] = gc1[np.arange(bs), ii].unsqueeze(1)
        x = self.drop(x)
        if self.shift:
            x = torch.roll(x, self.in_features, 1)
        x = self.fc2(x)
        gc2, indices2 = group_interconnect(x, self.groups)
        gc2 = self.group_c2(gc2)
        for ii, index in enumerate(indices2):
            x[:, index] = gc2[np.arange(bs), ii].unsqueeze(1)
        # x = self.bn2(x)
        y = x
        weight_soft = self.weight_soft(x)
        x = weight_soft[:, 0].view(-1, 1, 1, 1) * x + weight_soft[:, 1].view(-1, 1, 1, 1) * attn
        x = self.drop(x)
        
        return x
