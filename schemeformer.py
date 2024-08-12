# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MetaFormer implementation with hybrid stages
"""
import imp
from typing import Sequence, collections
from functools import partial, reduce
import torch
import torch.nn as nn
import os
import copy

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model


from misc_heads import Mlp
from mlp_heads import GroupNorm, PatchEmbed, LayerNormChannel, GroupLiteAttMlp, PartialGroupLiteAttMlp, GroupLiteSEMlp, ShuffleGroupLiteAttMlp, GroupLiteConvMlp, GroupLiteGCTAttMlp

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }

AttentionLayers = 'mlp'

class AddPositionEmb(nn.Module):
    """Module to add position embedding to input features
    """
    def __init__(
        self, dim=384, spatial_shape=[14, 14],
        ):
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = [spatial_shape]
        assert isinstance(spatial_shape, Sequence), \
            f'"spatial_shape" must by a sequence or int, ' \
            f'get {type(spatial_shape)} instead.'
        if len(spatial_shape) == 1:
            embed_shape = list(spatial_shape) + [dim]
        else:
            embed_shape = [dim] + list(spatial_shape)
        self.pos_embed = nn.Parameter(torch.zeros(1, *embed_shape))
    def forward(self, x):
        return x+self.pos_embed


class AddPositionEmbV2(nn.Module):
    """Module to add position embedding to input features
    """
    def __init__(
        self, dim=384, spatial_shape=[14, 14],
        ):
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = [spatial_shape]
        assert isinstance(spatial_shape, Sequence), \
            f'"spatial_shape" must by a sequence or int, ' \
            f'get {type(spatial_shape)} instead.'
        if len(spatial_shape) == 1:
            embed_shape = list(spatial_shape) + [dim]
        else:
            embed_shape = [dim] + list(spatial_shape)
        self.pos_embed = nn.Parameter(torch.zeros(1, *embed_shape))
       
    def forward(self, x):
        shape = x.shape
        upsample = nn.Upsample([shape[2], shape[3]])
        pos_embed = upsample(self.pos_embed)
        # print(x.shape, pos_embed.shape)
        return x+pos_embed


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        pool = self.pool(x)
        out = pool - x
        self.q, self.k, self.v = x, pool, out
        return out
    

class Attention(nn.Module):
    """Attention module that can take tensor with [B, N, C] or [B, C, H, W] as input.
    Modified from: 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, dim, head_dim=32, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % head_dim == 0, 'dim should be divisible by head_dim'
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        shape = x.shape
        if len(shape) == 4:
            B, C, H, W = shape
            N = H * W
            x = torch.flatten(x, start_dim=2).transpose(-2, -1) # (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        self.q, self.k, self.v = q, k, v

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    --dim: embedding dim
    --token_mixer: token mixer module
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, 
                 token_mixer=nn.Identity, 
                 mlp_block=nn.Identity,
                 mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=LayerNormChannel, 
                 drop=0., drop_path=0., name=None,  use_cca=False,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = DyMlp3(in_features=dim, hidden_features=mlp_hidden_dim,
                    #    act_layer=act_layer, drop=drop)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, #name=name, 
                       act_layer=act_layer, drop=drop, use_cca=use_cca)
                    
        # self.mlp = DYModule(dim, dim, mlp_ratio)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            mlp_out = self.mlp(self.norm2(x))
            if isinstance(mlp_out, list) or isinstance(mlp_out, tuple):
                mlp_out, _ = mlp_out
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * mlp_out)
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            mlp_out = self.mlp(self.norm2(x))
            if isinstance(mlp_out, list) or isinstance(mlp_out, tuple):
                mlp_out, _ = mlp_out
            x = x + self.drop_path(mlp_out)
        return x


def basic_blocks(dim, index, layers, token_mixer=nn.Identity, mlp_block=nn.Identity, 
                 mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=LayerNormChannel, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        if index > 1:
            n = f'network.{2*index+1}.{block_idx}.mlp'
        else:
            n = f'network.{2*index}.{block_idx}.mlp'
        # print(n)
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        use_cca = False
        # if index == 1 and (block_idx >= 4 or block_idx == 5):# block_idx == 5 or 
            # use_cca = True
        blocks.append(MetaFormerBlock(
            dim, token_mixer=token_mixer, mlp_ratio=mlp_ratio, mlp_block=mlp_block,
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, name=n, use_cca=use_cca,
            layer_scale_init_value=layer_scale_init_value, 
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class MetaFormer(nn.Module):
    """
    MetaFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios: the embedding dims and mlp ratios for the 4 stages
    --token_mixers: token mixers of different stages
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    --add_pos_embs: position embedding modules of different stages
    """
    def __init__(self, layers, embed_dims=None, 
                 token_mixers=None, mlp_ratios=None, mlp_blocks=[Mlp, Mlp, Mlp, Mlp],
                 norm_layer=LayerNormChannel, act_layer=nn.GELU, 
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 downsamples=None, down_patch_size=3, down_stride=2, down_pad=1, 
                 add_pos_embs=None, init_cfg=None, pretrained=None,
                 drop_rate=0., drop_path_rate=0., fork_feat=False,
                 frozen_stages=-1,
                 use_layer_scale=True, layer_scale_init_value=1e-5, cosine_attn_loss=False,
                 **kwargs):

        super().__init__()


        self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.frozen_layers = frozen_stages
        self.cosine_attn_loss = cosine_attn_loss
        self.attention_maps = None

        # if self.cosine_attn_loss:
        #     # return_ids=True
        #     # if self.batch_idx % self.distill_every_n_step == 0:
        #     if self.attention_maps is None:
        #         self.register_tokenmap_hooks()
        #     else:
        #         self.reset_attention_maps()

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=3, embed_dim=embed_dims[0])
        if add_pos_embs is None:
            add_pos_embs = [None] * len(layers)
        if token_mixers is None:
            token_mixers = [nn.Identity] * len(layers)
        if mlp_blocks is None:
            mlp_blocks = [nn.Identity] * len(layers)
        # set the main block in network
        network = []
        for i in range(len(layers)):
            if add_pos_embs[i] is not None:
                network.append(add_pos_embs[i](embed_dims[i]))
            stage = basic_blocks(embed_dims[i], i, layers, 
                                 token_mixer=token_mixers[i], mlp_ratio=mlp_ratios[i], mlp_block=mlp_blocks[i],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, 
                        padding=down_pad, 
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )
        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()


        self.network = nn.ModuleList(network)
       
        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model 
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()
        
        self._freeze_stages()

    
    def _freeze_stages(self):
        if self.frozen_layers >= 0:
            if self.patch_embed:
                self.patch_embed.eval()
                for param in self.patch_embed.parameters():
                    param.requires_grad = False

            for i in range(0, self.frozen_layers + 1):
                for name, m in self.named_modules():
                    if f'network.{i}' in name:
                        # m = getattr(self, name)
                        m.eval()
                        for param in m.parameters():
                            param.requires_grad = False


    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # init for mmdetection or mmsegmentation by loading 
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            
            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:                
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        return x

    def forward(self, x):

        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        # for image classification
        cls_out = self.head(x.mean([-2, -1]))
        return cls_out

    
    def register_evaluation_hooks(self):
        r"""Function for registering hooks during evaluation.
        We mainly store activation maps averaged over queries.
        """
        self.forward_hooks = []

        def save_activations(activations, name, module, inp, out):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            # out[0] - final output of attention layer
            # out[1] - attention probability matrix
            # if 'attn2' in name:
            #     assert out[1].shape[-1] == 77
            activations[name].append(out)
            # else:
            #     assert out[1].shape[-1] != 77
        attention_dict = collections.defaultdict(list)
        for name, module in self.named_modules():
            leaf_name = name.split('.')[-1]
            if 'mlp' in leaf_name:
                # Register hook to obtain outputs at every attention layer.
                self.forward_hooks.append(module.register_forward_hook(
                    partial(save_activations, attention_dict, name)
                ))
        # attention_dict is a dictionary containing attention maps for every attention layer
        self.attention_maps = attention_dict

    def register_tokenmap_hooks(self):
        r"""Function for registering hooks during evaluation.
        We mainly store activation maps averaged over queries.
        """
        self.forward_hooks = []

        def save_activations(selfattn_maps, n_maps, name, module, inp, out):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            # out[0] - final output of attention layer
            # out[1] - attention probability matrices
            if name in n_maps:
                n_maps[name] += 1
            else:
                n_maps[name] = 1

            if AttentionLayers in name and n_maps[name] > 0:
            # if n_maps[name] > 0:
                # print(name)
                if name in selfattn_maps:
                    selfattn_maps[name] += [out[1]]#.detach().cpu()]#[1:2]
                else:
                    selfattn_maps[name] = [out[1]]#.detach().cpu()]#[1:2]

        selfattn_maps = collections.defaultdict(list)
        n_maps = collections.defaultdict(list)

        for name, module in self.named_modules():
            leaf_name = name.split('.')[-1]
            if 'mlp' in leaf_name:
                # Register hook to obtain outputs at every attention layer.
                self.forward_hooks.append(module.register_forward_hook(
                    partial(save_activations, selfattn_maps, n_maps, name)
                ))
        # attention_dict is a dictionary containing attention maps for every attention layer
        self.selfattn_maps = selfattn_maps
        self.n_maps = n_maps

    def remove_tokenmap_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        self.selfattn_maps = None
        # self.crossattn_maps = None
        self.n_maps = None

    def reset_attention_maps(self):
        r"""Function to reset attention maps.
        We reset attention maps because we append them while getting hooks
        to visualize attention maps for every step.
        """
        for key in self.selfattn_maps:
            self.selfattn_maps[key] = []
        # for key in self.crossattn_maps:
        #     self.crossattn_maps[key] = []

model_urls = {
    "metaformer_id_s12": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_id_s12.pth.tar",
    "metaformer_pppa_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_pppa_s12_224.pth.tar",
    "metaformer_ppaa_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_ppaa_s12_224.pth.tar",
    "metaformer_pppf_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_pppf_s12_224.pth.tar",
    "metaformer_ppff_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_ppff_s12_224.pth.tar",
}

@register_model
def schemeformer_ppaa_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    add_pos_embs = [None, None, 
        partial(AddPositionEmb, spatial_shape=[14, 14]), None]
    token_mixers = [Pooling, Pooling, Attention, Attention]
    mlp_blocks = [Mlp, Mlp, Mlp, Mlp]
    mlp_ratios = [8, 8, 8, 8]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_blocks=mlp_blocks,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        add_pos_embs=add_pos_embs,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['schemeformer_ppaa_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def schemeformer_ppaa_s36_224(pretrained=False, **kwargs):
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    add_pos_embs = [None, None, 
        partial(AddPositionEmb, spatial_shape=[14, 14]), None]
    token_mixers = [Pooling, Pooling, Attention, Attention]
    mlp_blocks = [Mlp, Mlp, Mlp, Mlp]
    mlp_ratios = [8, 8, 8, 8]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_blocks=mlp_blocks,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        add_pos_embs=add_pos_embs,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['schemeformer_ppaa_s36_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def grouporgsemlpformer_ppaa_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    add_pos_embs = [None, None, 
        partial(AddPositionEmb, spatial_shape=[14, 14]), None]
    token_mixers = [Pooling, Pooling, Attention, Attention]
    mlp_ratios = [8, 8, 8, 8]
    mlp_blocks = [GroupLiteSEMlp, GroupLiteSEMlp, GroupLiteSEMlp, GroupLiteSEMlp]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_blocks=mlp_blocks,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        add_pos_embs=add_pos_embs,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['grouporgsemlpformer_ppaa_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def shufflegrouporgattnmlpformer_ppaa_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    add_pos_embs = [None, None, 
        partial(AddPositionEmb, spatial_shape=[14, 14]), None]
    token_mixers = [Pooling, Pooling, Attention, Attention]
    mlp_ratios = [8, 8, 8, 8]
    mlp_blocks = [ShuffleGroupLiteAttMlp, ShuffleGroupLiteAttMlp, ShuffleGroupLiteAttMlp, ShuffleGroupLiteAttMlp]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_blocks=mlp_blocks,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        add_pos_embs=add_pos_embs,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['shufflegrouporgattnmlpformer_ppaa_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model
    

@register_model
def groupliteconvmlpformer_ppaa_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    add_pos_embs = [None, None, 
        partial(AddPositionEmb, spatial_shape=[14, 14]), None]
    token_mixers = [Pooling, Pooling, Attention, Attention]
    mlp_ratios = [8, 8, 8, 8]
    mlp_blocks = [GroupLiteConvMlp]*4
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_blocks=mlp_blocks,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        add_pos_embs=add_pos_embs,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['groupliteattnmlpformer_ppaa_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def groupliteattnmlpformer_ppaa_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    add_pos_embs = [None, None, 
        partial(AddPositionEmb, spatial_shape=[14, 14]), None]
    token_mixers = [Pooling, Pooling, Attention, Attention]
    mlp_ratios = [8, 8, 8, 8]
    mlp_blocks = [GroupLiteGCTAttMlp, GroupLiteGCTAttMlp, GroupLiteGCTAttMlp, GroupLiteGCTAttMlp]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_blocks=mlp_blocks,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        add_pos_embs=add_pos_embs,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['groupliteattnmlpformer_ppaa_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


if has_mmseg and has_mmdet:
    """
    The following models are for dense prediction based on 
    mmdetection and mmsegmentation
    """
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class grouporgattnmlpformer_ppaa_s12_224_feat(MetaFormer):
        """
        model, Params: 12M
        """
        def __init__(self, **kwargs):
            layers = [2, 2, 6, 2]
            embed_dims = [64, 128, 320, 512]
            add_pos_embs = [None, None, None, None]
                        # partial(AddPositionEmbV2, spatial_shape=[14, 14]), None]
            token_mixers = [Pooling, Pooling, Attention, Attention]
            mlp_ratios = [8, 8, 8, 8]
            mlp_blocks = [GroupLiteAttMlp, GroupLiteAttMlp, GroupLiteAttMlp, GroupLiteAttMlp]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims, add_pos_embs=add_pos_embs, norm_layer=GroupNorm,
                mlp_ratios=mlp_ratios, downsamples=downsamples, token_mixers=token_mixers,
                fork_feat=True, mlp_blocks=mlp_blocks, num_classes=80,
                **kwargs)
    