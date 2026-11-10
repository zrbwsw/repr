# --------------------------------------------------------
# References:
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

# 中文导读：原始 JiT 所用的 RoPE、RMSNorm 与固定 sin/cos 位置编码工具。
# RePR 在 model_repr.py 内另有支持 mask、位置索引和 register 位置的实现；
# 本文件的 Fast RoPE 对前缀 token 不旋转，不能据此推断 RePR 的 register 行为。
from math import pi

import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat


# 先在非拼接维度广播到共同尺寸，再沿 dim 拼接。
# 主要用于把高度方向和宽度方向的一维频率展开为二维网格。
def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


# 把最后一维每相邻两个通道 (a,b) 变成 (-b,a)，表示二维平面上的 90 度旋转。
# 与 cos/sin 权重相加后构成 RoPE 的逐对旋转；不是把前后两大半段交换。
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


# 保留二维网格形状的 RoPE 版本，可从指定特征维度 start_index 开始旋转。
# 预计算的 cos/sin 用 register_buffer 保存，属于非学习参数，会随模块移动设备。
class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        # 坐标按 ft_seq_len 归一化后映射到 pt_seq_len 的参考尺度；本模块构造时
        # 决定该映射，而非 forward 每次根据输入长度动态重建。h/w 分别占一组通道。
        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    # 只旋转选定的 rot_dim 个通道，左右未选部分原样拼回。
    # 调用方的空间轴与预计算二维 cos/sin 网格需要可广播匹配。
    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim = -1)


# 将二维网格展平为 [N_img,head_dim]，便于直接处理 [B,heads,N,head_dim]。
# 该旧版本把 cos/sin 直接作为 CUDA Tensor 属性保存，并非 Parameter 或注册 buffer；
# 它依赖构造时的当前 CUDA 设备，和本文件上方普通版的设备/保存行为不同。
class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        num_cls_token = 0
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])  # [N_img, D]
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # 前缀 token 使用单位旋转：cos=1、sin=0，因此保持其 Q/K 值不变。
            # 图像 token 仍保留原本的二维位置序列，前缀不会改变其网格坐标。
            # prepend in-context cls token
            N_img, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

            self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).cuda()  # [N_cls+N_img, D]
            self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).cuda()
        else:
            self.freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()
            self.freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()

    # 输入最后两维必须匹配构造时预计算的 token 数和旋转通道数。
    # RoPE 在 Q/K 上应用，不是把一个位置向量加到 token 特征上。
    def forward(self, t): return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin


# 按最后一维的均方根归一化，再乘逐通道可学习 weight；不减均值，也没有 bias。
# 统计量在 FP32 中计算，最后转回输入 dtype，减少低精度平方/求和误差。
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# 生成固定二维绝对位置表 [grid_size^2,embed_dim]。
# 若启用 cls_token 且 extra_tokens>0，前置对应数量的全零位置行。
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    # 保留 meshgrid 的原始坐标顺序：grid[0] 是宽度坐标，grid[1] 是高度坐标。
    # 下游 emb_h/emb_w 的变量命名不会改变这里实际传入的坐标轴。
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


# 把总维度一分为二，分别编码 grid 的两个坐标分量后拼接。
# 每个一维编码还要分 sin/cos，因此常用配置要求 embed_dim 能被 4 整除。
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


# 位置与按指数衰减的频率做外积，前半维为 sin，后半维为 cos。
# 本函数使用 NumPy 产生固定表，不建立训练梯度；调用方再转成 Torch Tensor。
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb