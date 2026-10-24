# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from slop.augmentation import post_crop_augment_batch, random_resized_crop_batch
from slop.augmentation_overlap import visible_a_to_d_patch_overlap


MUP_BASE_WIDTH = 768


class MuPLinear(nn.Linear):
    """Marks a hidden or output linear weight for MuP LR scaling."""


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


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',  # GIGA TODO why not set?
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
        
        self.num_cls_token = num_cls_token
        self.register_buffer('freqs_cos', freqs.cos().view(-1, freqs.shape[-1]), persistent=False)
        self.register_buffer('freqs_sin', freqs.sin().view(-1, freqs.shape[-1]), persistent=False)

    def forward(self, t, m=None, use_cls=True, idx=None, m_selected=None, cls_idx=None):
        B, H, N, D = t.shape
        # two changes
        # 1: Subindex for m
        # 2: Legacy CLS padding reuses a selected token position unless cls_idx is given
        # 3: idx: B, N
        # 4: m_selected: B, N_selected; for CLS, copy the first unmasked selected token
        # 5: cls_idx: B; for v14, put CLS at the visible-image center

        if idx is not None:
            freqs_cos = self.freqs_cos.expand(B, -1, -1).gather(1, idx.unsqueeze(-1).expand(-1,-1,D)).view(B, 1, -1, D)
            freqs_sin = self.freqs_sin.expand(B, -1, -1).gather(1, idx.unsqueeze(-1).expand(-1,-1,D)).view(B, 1, -1, D)
        elif m is not None:
            freqs_cos = self.freqs_cos.expand(B, -1, -1)[~m].view(B, 1, -1, D)
            freqs_sin = self.freqs_sin.expand(B, -1, -1)[~m].view(B, 1, -1, D)
        else:
            freqs_cos = self.freqs_cos.view(1, 1, -1, D).expand(B, 1, -1, D)
            freqs_sin = self.freqs_sin.view(1, 1, -1, D).expand(B, 1, -1, D)

        if use_cls and self.num_cls_token > 0:
            if cls_idx is not None:
                cls_idx = cls_idx.view(B, 1, 1).expand(B, 1, D)
                cls_cos = self.freqs_cos.expand(B, -1, -1).gather(1, cls_idx).view(B, 1, 1, D)
                cls_sin = self.freqs_sin.expand(B, -1, -1).gather(1, cls_idx).view(B, 1, 1, D)
            elif m_selected is None:
                cls_cos = freqs_cos[:, :, :1, :]
                cls_sin = freqs_sin[:, :, :1, :]
            else:
                cls_idx = m_selected.int().argmin(dim=1)
                cls_idx = cls_idx.view(B, 1, 1, 1).expand(B, 1, 1, D)
                cls_cos = freqs_cos.gather(2, cls_idx)
                cls_sin = freqs_sin.gather(2, cls_idx)
            cls_cos = cls_cos.expand(B, 1, self.num_cls_token, D)
            cls_sin = cls_sin.expand(B, 1, self.num_cls_token, D)
            freqs_cos = torch.cat([cls_cos, freqs_cos], dim=2)
            freqs_sin = torch.cat([cls_sin, freqs_sin], dim=2)

        return  t * freqs_cos + rotate_half(t) * freqs_sin

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            MuPLinear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0, attn_mask=None) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query, 
        key, 
        value, 
        attn_mask=attn_mask, 
        dropout_p=dropout_p, 
        is_causal=False
    )


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = MuPLinear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = MuPLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0., attn_mask=attn_mask)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.q = MuPLinear(dim, dim, bias=qkv_bias)
        self.kv = MuPLinear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = MuPLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, q_rope, k_rope, attn_mask=None):
        B, N, C = x.shape
        K = context.size(1)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        kv = self.kv(context).reshape(B, K, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q_rope(q)
        k = k_rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0., attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = MuPLinear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = MuPLinear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = MuPLinear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            MuPLinear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            MuPLinear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x,  c, feat_rope=None, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope, attn_mask=attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
class MAGEBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    @torch.compile(dynamic=False)
    def forward(self, x, feat_rope=None, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope=feat_rope, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class RepresentationCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm_q = RMSNorm(hidden_size, eps=1e-6)
        self.norm_context = RMSNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                         attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    @torch.compile
    def forward(self, x, context, q_rope, k_rope, attn_mask=None):
        x = x + self.cross_attn(
            self.norm_q(x),
            self.norm_context(context),
            q_rope,
            k_rope,
            attn_mask=attn_mask,
        )
        x = x + self.mlp(self.norm2(x))
        return x


def modulate_1d(x, shift, scale):
    return x * (1 + scale) + shift

class FinalLayerMARMLP(nn.Module):
    """
    The final layer for MAR MLP adopted from DiT. 
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = MuPLinear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            MuPLinear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate_1d(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FinalLayerLearnedModulation(nn.Module):
    """
    Final layer with learned AdaLN-style shift and scale.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.shift = nn.Parameter(torch.zeros(1, 1, model_channels), requires_grad=True)
        self.scale = nn.Parameter(torch.zeros(1, 1, model_channels), requires_grad=True)
        self.linear = MuPLinear(model_channels, out_channels, bias=True)

    def forward(self, x):
        x = modulate_1d(self.norm_final(x), self.shift, self.scale)
        x = self.linear(x)
        return x

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels,
        use_swiglu=False,
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        if use_swiglu:
            self.mlp = SwiGLUFFN(channels, 4 * channels)
        else:
            self.mlp = nn.Sequential(
                MuPLinear(channels, channels, bias=True),
                nn.SiLU(),
                MuPLinear(channels, channels, bias=True),
            )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            MuPLinear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate_1d(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        use_swiglu=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = MuPLinear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
                use_swiglu=use_swiglu,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayerMARMLP(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                if module is self.input_proj:
                    nn.init.uniform_(
                        module.weight,
                        -math.sqrt(3 / self.in_channels),
                        math.sqrt(3 / self.in_channels),
                    )
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(
            self.time_embed.mlp[2].weight,
            std=0.02 * math.sqrt(MUP_BASE_WIDTH / self.model_channels),
        )

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @torch.compile
    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)


class REPR(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        condition_on_class=False,
        in_context_len=16,
        in_context_start=8,
        repr_dim=768,
        num_patch_heads=4,
        decoder_depth=6,
        diff_mlp_width=1024,
        diff_mlp_depth=3,
        diff_mlp_swiglu=False,
        batch_repeat=1,
        batch_repeat_mar=1,
        diff_repeat=4,
        d_num_p=8,
        diff_repeat_repr=32,
        d_max=48,
        mask_m=256 - 76,
        pmi_target_mul=1,
        source_time_repeat=1,
        num_repr_heads=1,
        masked_selfdistill_mask_ratio=0.75,
        masked_selfdistill_loss_weight=1.0,
        repr_loss_weight=0.1,
        repr_delta=0.2,
        imagenet_loss_weight=0.0,
        pad_dropout=0.05,
        repr_noise_bins=8,
        add_input_noise=False,
        mu_repr=-1.2,
        std_repr=1.2,
        mu_mar=-1.2,
        std_mar=1.2,
        version='8',
        pack_encoder_DADp=None,
        pack_encoder_AD=None,
        mask_strategy_mar='block',
        mask_strategy_repr='block',
        do_simclr_augs=True,
        noise_a=False,
        is_first_iter=True,
        is_rope_bug_fixed=False,
        is_repr_encoderdecoder=False,
        is_raw_pmi=False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        self.y_embedder = LabelEmbedder(num_classes, hidden_size) if condition_on_class else None
        self.repr_dim = repr_dim
        self.num_patch_heads = num_patch_heads
        self.batch_repeat = batch_repeat
        self.batch_repeat_mar = batch_repeat_mar
        self.diff_repeat = diff_repeat
        self.version = version
        self.mask_m = mask_m
        self.masked_selfdistill_mask_ratio = masked_selfdistill_mask_ratio
        self.masked_selfdistill_loss_weight = masked_selfdistill_loss_weight
        self.repr_loss_weight = repr_loss_weight
        self.repr_delta = repr_delta
        self.imagenet_loss_weight = imagenet_loss_weight
        self.pad_dropout = pad_dropout
        self.pack_encoder_DADp = self.num_patches if pack_encoder_DADp is None else pack_encoder_DADp
        self.pack_encoder_AD = self.num_patches if pack_encoder_AD is None else pack_encoder_AD
        self.mask_strategy_mar = mask_strategy_mar
        self.mask_strategy_repr = mask_strategy_repr
        self.do_simclr_augs = do_simclr_augs
        self.noise_a = noise_a
        self.is_first_iter = is_first_iter
        self.is_repr_encoderdecoder = is_repr_encoderdecoder
        self.is_raw_pmi = is_raw_pmi
        self.mu_repr, self.std_repr, self.mu_mar, self.std_mar = mu_repr, std_repr, mu_mar, std_mar

        # linear embed
        self.x_embedder = nn.Linear(self.patch_dim, hidden_size, bias=True)

        if self.in_context_len > 0:
            self.in_context_tokens = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_tokens, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # transformer
        self.encoder = nn.ModuleList([
            MAGEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
            for i in range(depth)
        ])

        self.decoder = nn.ModuleList([
            MAGEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
            for i in range(decoder_depth)
        ])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        torch.nn.init.normal_(self.mask_token, std=.02)

        if version == '18':
            print('please work repr cross attention learned modulation pad dropout and multinoise')
            self.d_num_p = d_num_p
            self.d_max = d_max
            self.hw_len = input_size // patch_size
            self.pmi_target_mul = pmi_target_mul
            self.source_time_repeat = source_time_repeat
            self.num_repr_heads = num_repr_heads
            self.repr_noise_bins = repr_noise_bins
            self.diff_bin_repeat_repr = diff_repeat_repr // repr_noise_bins
            self.add_input_noise = add_input_noise

            self.factor_pad = 2
            self.pad_hw = self.hw_len + 2*self.factor_pad
            pad_dropout_mask = torch.zeros(self.pad_hw, self.pad_hw)
            pad_dropout_mask[self.factor_pad:self.factor_pad+self.hw_len,self.factor_pad:self.factor_pad+self.hw_len] = 1
            self.register_buffer('pad_dropout_mask', pad_dropout_mask.flatten()[...,None])
            self.factor_posembed = nn.Parameter(torch.zeros((hw_seq_len*2-1+2*self.factor_pad)**2, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.factor_posembed, std=.02)

        if version == '22':
            print('please work repr joint encoding')
            self.d_num_p = d_num_p
            self.d_max = d_max
            self.hw_len = input_size // patch_size
            self.pmi_target_mul = pmi_target_mul
            self.source_time_repeat = source_time_repeat
            self.num_repr_heads = num_repr_heads
            self.repr_noise_bins = repr_noise_bins
            self.diff_bin_repeat_repr = diff_repeat_repr // repr_noise_bins

            self.factor_pad = 1
            self.pad_hw = self.hw_len + 2*self.factor_pad
            pad_dropout_mask = torch.zeros(self.pad_hw, self.pad_hw)
            pad_dropout_mask[self.factor_pad:self.factor_pad+self.hw_len,self.factor_pad:self.factor_pad+self.hw_len] = 1
            self.register_buffer('pad_dropout_mask', pad_dropout_mask.flatten()[...,None])
            self.factor_posembed = nn.Parameter(torch.zeros((hw_seq_len*2-1+2*self.factor_pad)**2, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.factor_posembed, std=.02)
            self.feat_rope_incontext_pad = VisionRotaryEmbeddingFast(
                dim=half_head_dim * (1 if is_rope_bug_fixed else 2),  # GIGA TODO SHOOT BUG (NOT TECHNICALLY INVALID) WHEN DID THIS OCCUR CANT REMOVE WITHOUT INVALIDATING CKPTS
                pt_seq_len=hw_seq_len + 2*self.factor_pad,
                num_cls_token=self.in_context_len
            )
            self.repr_final_layer = FinalLayerLearnedModulation(hidden_size, repr_dim)
            self.repr_final_layer_global = FinalLayerLearnedModulation(hidden_size, repr_dim)
            self.representation_encoder = nn.ModuleList([
                RepresentationCrossAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                                                  attn_drop=attn_drop, proj_drop=proj_drop)
                for _ in range(3)
            ])

            self.register_buffer('noise_bin_limits', torch.linspace(0,1,self.repr_noise_bins+1))

            self.is_d_embed = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.is_d_embed, std=.02)

            if self.is_repr_encoderdecoder:
                self.repr_encoder = nn.ModuleList([
                    MAGEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
                    for i in range(3)
                ])

            if self.is_first_iter:
                self.diff_mlp = SimpleMLPAdaLN(self.patch_dim, hidden_size, self.patch_dim, hidden_size, diff_mlp_depth, use_swiglu=diff_mlp_swiglu)
            else:
                self.feats_embedder = nn.Linear(self.repr_dim, hidden_size, bias=True)
                self.diff_mlp = SimpleMLPAdaLN(self.repr_dim, hidden_size, self.repr_dim, hidden_size, diff_mlp_depth, use_swiglu=diff_mlp_swiglu)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize the patch embedding linear layers.
        nn.init.uniform_(
            self.x_embedder.weight,
            -math.sqrt(3 / self.patch_dim),
            math.sqrt(3 / self.patch_dim),
        )
        if self.x_embedder.bias is not None:
            nn.init.constant_(self.x_embedder.bias, 0)

        self.diff_mlp.initialize_weights()
        if hasattr(self, "diff_mlp_feats"):
            self.diff_mlp_feats.initialize_weights()
        if hasattr(self, "repr_final_layer_noisebin"):  # xavier for noise_bins batched linear layer
            fan_in = self.repr_final_layer_noisebin.linear.in_features
            fan_out = self.repr_final_layer_noisebin.linear.out_features // self.repr_noise_bins
            nn.init.uniform_(self.repr_final_layer_noisebin.linear.weight, -math.sqrt(6.0 / (fan_in + fan_out)), math.sqrt(6.0 / (fan_in + fan_out)))
            if self.repr_final_layer_noisebin.linear.bias is not None:
                nn.init.constant_(self.repr_final_layer_noisebin.linear.bias, 0)
        # self.patch_repr_mlp.initialize_weights()

    def get_mup_param_groups(self, weight_decay=0.0):
        """AdamW groups for MuP transfer from the width-768 REPR-B model."""
        width_mult = self.hidden_size / MUP_BASE_WIDTH
        matrix_weights = {
            id(module.weight)
            for module in self.modules()
            if isinstance(module, MuPLinear)
        }
        groups = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = param.ndim == 1 or name.endswith(".bias")
            lr_scale = 1.0 / width_mult if id(param) in matrix_weights else 1.0
            groups.setdefault((lr_scale, no_decay), []).append(param)

        return [
            {
                "params": groups[(lr_scale, no_decay)],
                "lr_scale": lr_scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
            for lr_scale, no_decay in sorted(
                groups,
                key=lambda key: (not key[1], key[0] != 1.0),
            )
        ]

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, c, p, p))
        x = torch.einsum('nhwcpq->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def patchify(self, x):
        """
        x: (B, C, H, W)
        returns: (B, N, patch_size**2 * C)
        """
        B, C, H, W = x.shape
        assert H == self.input_size and W == self.input_size, (
            f"Input image size ({H}*{W}) doesn't match model ({self.input_size}*{self.input_size})."
        )
        p = self.patch_size
        h = H // p
        w = W // p
        return x.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * p * p)
    
    def get_attention_mask(self, m, attn_mode: Literal["oneway", "separate", "fullseq"]="oneway"):
        k = m.unsqueeze(1)
        q = m.unsqueeze(2)
        if attn_mode == "oneway":
            attn_mask = (k == 0) | (q == k)
        elif attn_mode == "separate":
            attn_mask = (q == k)
        elif attn_mode == "fullseq":
            attn_mask = torch.ones_like(q == k)

        return attn_mask.unsqueeze(1)
    
    def sample_m(self, B, N, device):
        mask_ratio = 0.75
        mask_n = int(N * mask_ratio)
        all_idx = torch.rand((B,N), device=device).argsort(dim=1)
        is_mask = torch.zeros((B,N), dtype=bool, device=device)
        is_mask.scatter_(1, all_idx[:,:mask_n], 1)
        m = torch.where(
            is_mask,
            torch.arange(1, N+1, device=device).expand(B,-1), # could just be binary because mar loss
            0,
        )
        return m
    
    def sample_m_mar(self, B, N, device, min_mask_ratio = 0.7):
        # Guarantees 
        # m_index 1 has min_mask_ratio selected
        # m_index 2 ~ U[min_mask_ratio,1] per batch index masked. 
        #   Global sum equal for compile

        mask_m = min(self.mask_m, N)
        min_mask_ratio = mask_m / N
        mask_values = torch.rand((B,N), device=device)
        mask_ps = torch.rand((B,1), device=device) * (1-min_mask_ratio) + min_mask_ratio  #U[0.7,1.0]

        all_idx = mask_values.argsort(dim=1)
        is_mask_idx = torch.zeros((B,N), dtype=bool, device=device)
        is_mask_idx.scatter_(1, all_idx[:,:mask_m], 1)

        mask_values_2 = torch.where(mask_values < mask_ps, mask_values * 0.5 / mask_ps, 0.5 + 0.5*(mask_values-mask_ps)/(1-mask_ps+1e-5))
        mask_values_2 = torch.where(mask_values_2 <= mask_values_2.topk(dim=-1, k=mask_m, largest=False).values[:,-1:], -1, mask_values_2) # a.s. no bug
        mask_n_2 = int(B * N * (1-(1-min_mask_ratio)/2))
        all_idx_2 = mask_values_2.view(-1).argsort(dim=0)
        is_mask_stage_2 = torch.zeros((B*N), dtype=bool, device=device)
        is_mask_stage_2.scatter_(0, all_idx_2[:mask_n_2], True)
        is_mask_stage_2 = is_mask_stage_2.view(B,N)

        is_mask_stage_1 = torch.where(is_mask_idx, torch.ones((B,N), device=device, dtype=bool), False)
        return is_mask_stage_1, is_mask_stage_2
    
    def sample_m_reprv2(self, B_IN, N, device, min_mask_ratio = 0.7):
        # Guarantees 
        # m_index 1 has min_mask_ratio selected
        # m_index 2 ~ U[min_mask_ratio,1] per batch index masked. 
        #   Global sum equal for compile
        # is_mask_background_2: background without D, shape b,n
        # is_mask_background_2_dp: background without DP, exactly 1 fewer true per element, shape d,b,n
        # is_mask_background_1_dp: background without DP MAE encoder mask  TODO clever 2x speedup.
        # GIGA TODO: DECIDE: S also shared like R for amortizations or S averages over D for complete variance reduction. SHARED --> DP-D REPR THE SAME

        R = self.batch_repeat
        P = self.d_num_p

        is_selected_encoder, is_mask = self.sample_m_mar(R*B_IN,N,device,min_mask_ratio); is_selected_encoder = ~is_selected_encoder

        is_dp_idx = (repeat(~is_mask, '(R B) N -> R B N', R=R).sum(dim=0) + torch.rand((B_IN,N), device=device) * 0.5).argsort(dim=1)[:,:P]
        is_dp = torch.zeros(P*B_IN, N, device=device, dtype=bool).scatter(1, rearrange(is_dp_idx, 'b p -> (p b) 1'), 1)
        is_dp = rearrange(is_dp, '(P B) N -> P B N', P=P)

        d_max = min(self.d_max, N)
        k = torch.randint(0, d_max + 1, (B_IN, 1), device=device)  # how many False per row
        is_background_idx = (torch.rand(B_IN, N, device=device) + is_dp.any(0)).argsort(dim=1)[:, :d_max]  # 0% chance take is_dp
        is_mask_background = torch.ones(B_IN, N, dtype=torch.bool, device=device).scatter(1, is_background_idx, torch.arange(d_max, device=device).unsqueeze(0) >= k) #| True  # GIGA TODO
        is_mask_background_dp = repeat(is_mask_background, 'B N -> P B N', P=P) & (~is_dp)
        is_selected_background_dp = torch.zeros(P, B_IN, N, dtype=bool, device=device).scatter(2, is_mask_background_dp.int().argsort(2)[:,:,:d_max+1], True)
        
        is_selected_background = is_selected_background_dp[0]  # Any element works
        is_selected_background_dp = rearrange(is_selected_background_dp, 'P B N -> (P B) N')
        is_mask_background_dp = rearrange(is_mask_background_dp, 'P B N -> (P B) N')

        return is_selected_encoder, is_mask, is_selected_background, is_mask_background, is_selected_background_dp, is_mask_background_dp, is_dp

    
    def sample_m_reprv2_1(self, B_IN, N, device, H=16, W=16, min_mask_ratio = 0.7):
        # Guarantees 
        # m_index 1 has min_mask_ratio selected
        # m_index 2 ~ U[min_mask_ratio,1] per batch index masked. 
        #   Global sum equal for compile
        # is_mask_background_2: background without D, shape b,n
        # is_mask_background_2_dp: background without DP, exactly 1 fewer true per element, shape d,b,n
        # is_mask_background_1_dp: background without DP MAE encoder mask  TODO clever 2x speedup.
        # GIGA TODO: DECIDE: S also shared like R for amortizations or S averages over D for complete variance reduction. SHARED --> DP-D REPR THE SAME

        
        def block_mask(B, N, M, block=4):
            # Slop code that works
            assert H * W == N and 0 <= M <= N and block >= 1

            self.mask_m

            # balanced counts over [M, N], exact total B * (N + M) // 2
            L = N - M + 1
            q, r = divmod(B, L)
            levels = M + torch.arange(L, device=device)
            counts = levels.repeat(q)

            if r:
                target = B * (N + M) // 2
                rem = target - q * levels.sum()
                base = torch.arange(r, device=device)
                extra = rem - (r * M + r * (r - 1) // 2)
                counts = torch.cat([
                    counts,
                    M + base + extra // r + (base >= r - extra % r).long()
                ])

            counts = counts[torch.randperm(B, device=device)]

            # blocky random priority field
            p0, p1 = (block - 1) // 2, block // 2
            x = torch.rand(B, 1, H, W, device=device)
            scores = F.max_pool2d(
                F.pad(x, (p0, p1, p0, p1), mode="circular"),
                kernel_size=block,
                stride=1,
            ).flatten(1)

            scores = scores + 1e-6 * torch.rand_like(scores)
            rank = scores.argsort(dim=1, descending=True).argsort(dim=1)

            return rank < counts[:, None]

        R = self.batch_repeat
        P = self.d_num_p

        is_mask = block_mask(R*B_IN,N,self.mask_m)

        is_valid_mask = rearrange(is_mask, 'RB (H W) -> RB H W', H=H, W=W).clone()
        is_valid_target = is_valid_mask.clone()
        # is_valid_target[:,:-1,:]    &= is_valid_mask[:,1:,:]
        # is_valid_target[:,:,:-1]    &= is_valid_mask[:,:,1:]
        # is_valid_target[:,1:,:]     &= is_valid_mask[:,:-1,:]
        # is_valid_target[:,:,1:]     &= is_valid_mask[:,:,:-1]
        # is_valid_target[:,:-1,:-1]  &= is_valid_mask[:,1:,1:]
        # is_valid_target[:,:-1,1:]   &= is_valid_mask[:,1:,:-1]
        # is_valid_target[:,1:,:-1]   &= is_valid_mask[:,:-1,1:]
        # is_valid_target[:,1:,1:]    &= is_valid_mask[:,:-1,:-1]
        # is_valid_target[:,:-2,:]    &= is_valid_mask[:,2:,:]
        # is_valid_target[:,:,:-2]    &= is_valid_mask[:,:,2:]
        # is_valid_target[:,2:,:]     &= is_valid_mask[:,:-2,:]
        # is_valid_target[:,:,2:]     &= is_valid_mask[:,:,:-2]
        is_valid_target = rearrange(is_valid_target, 'RB H W -> RB (H W)')

        is_dp_idx = (repeat(~is_valid_target, '(R B) N -> R B N', R=R).sum(dim=0) + torch.rand((B_IN,N), device=device) * 0.5).argsort(dim=1)[:,:P]
        is_dp = torch.zeros(P*B_IN, N, device=device, dtype=bool).scatter(1, rearrange(is_dp_idx, 'b p -> (p b) 1'), 1)
        is_dp = rearrange(is_dp, '(P B) N -> P B N', P=P)

        d_max = min(self.d_max, N)
        k = torch.randint(0, d_max + 1, (B_IN, 1), device=device)  # how many False per row
        is_background_idx = (torch.rand(B_IN, N, device=device) + is_dp.any(0)).argsort(dim=1)[:, :d_max]  # 0% chance take is_dp
        is_mask_background = torch.ones(B_IN, N, dtype=torch.bool, device=device).scatter(1, is_background_idx, torch.arange(d_max, device=device).unsqueeze(0) >= k)
        is_mask_background_dp = repeat(is_mask_background, 'B N -> P B N', P=P) & (~is_dp)
        is_selected_background_dp = torch.zeros(P, B_IN, N, dtype=bool, device=device).scatter(2, is_mask_background_dp.int().argsort(2)[:,:,:d_max+1], True)
        
        is_selected_background = is_selected_background_dp[0]  # Any element works
        is_selected_background_dp = rearrange(is_selected_background_dp, 'P B N -> (P B) N')
        is_mask_background_dp = rearrange(is_mask_background_dp, 'P B N -> (P B) N')

        is_mask = torch.cat([is_mask_background,is_mask],dim=0)  # GIGA TODO REQUIRES IS D < IS M
        is_selected_encoder = torch.zeros_like(is_mask).scatter(1, torch.rand_like(is_mask.float()).masked_fill(~is_mask, -1).argsort(dim=1)[:, : N - self.mask_m], True)
        is_valid_target = is_mask

        return is_selected_encoder, is_mask, is_selected_background, is_mask_background, is_selected_background_dp, is_mask_background_dp, is_dp, is_valid_target

    def _balanced_mask_counts(self, B, N, min_masked, device):
        min_masked = min(min_masked, N)
        levels = min_masked + torch.arange(N - min_masked + 1, device=device)
        q, r = divmod(B, levels.numel())
        counts = levels.repeat(q)

        if r:
            target = B * (N + min_masked) // 2
            rem = target - q * levels.sum()
            base = torch.arange(r, device=device)
            extra = rem - (r * min_masked + r * (r - 1) // 2)
            counts = torch.cat([
                counts,
                min_masked + base + extra // r + (base >= r - extra % r).long(),
            ])

        return counts[torch.randperm(B, device=device)]

    def sample_m_reprv2_1_random(self, B_IN, N, device, H=16, W=16, nonadjacent=0, R=1, check_A_overlap=True, **kwargs):
        assert H * W == N
        assert nonadjacent == 0

        P = self.d_num_p
        d_max = min(self.d_max, N)

        def random_mask(B, min_masked):
            counts = self._balanced_mask_counts(B, N, min_masked, device)
            rank = torch.rand(B, N, device=device).argsort(dim=1).argsort(dim=1)
            return rank < counts[:, None]

        is_mask_A = random_mask(R * B_IN, self.mask_m)
        is_mask_D = random_mask(B_IN, N - d_max)

        is_valid_A = rearrange(is_mask_A, '(R B) N -> R B N', R=R).all(dim=0)
        if check_A_overlap:
            is_valid_dp = is_valid_A & is_mask_D
        else:
            is_valid_dp = is_mask_D
        is_dp_idx = (torch.rand(B_IN, N, device=device) + is_valid_dp.float()).argsort(dim=1, descending=True)[:, :P]
        is_dp = torch.zeros(P * B_IN, N, device=device, dtype=bool).scatter(1, rearrange(is_dp_idx, 'B P -> (P B) 1'), True)
        is_dp = rearrange(is_dp, '(P B) N -> P B N', P=P)

        is_mask_Dp = repeat(is_mask_D, 'B N -> P B N', P=P) & (~is_dp)

        def get_selected(mask, num_selected):
            rank = torch.rand_like(mask.float()).masked_fill(~mask, -1).argsort(dim=1)
            return torch.zeros_like(mask).scatter(1, rank[:, :num_selected], True)

        num_selected_addp = max(N - self.mask_m, self.d_max + 1)
        is_selected_A = get_selected(is_mask_A, num_selected_addp)
        is_selected_D = get_selected(is_mask_D, num_selected_addp)
        is_selected_Dp = get_selected(rearrange(is_mask_Dp, 'P B N -> (P B) N'), num_selected_addp)
        is_mask_AD = is_mask_A & repeat(is_mask_D, 'B N -> (R B) N', R=R)
        is_selected_AD = get_selected(is_mask_AD, N - self.mask_m + self.d_max)
        is_valid_target = torch.cat([is_mask_D, is_mask_A], dim=0)

        return is_selected_A, is_mask_A, is_selected_D, is_mask_D, is_selected_Dp, rearrange(is_mask_Dp, 'P B N -> (P B) N'), is_selected_AD, is_mask_AD, is_dp, is_valid_target

    def sample_m_reprv2_1_square(self, B_IN, N, device, H=16, W=16, nonadjacent=1, R=1, **kwargs):
        assert H * W == N
        assert 0 <= nonadjacent < min(H, W)

        P = self.d_num_p

        def square_mask(B_, min_m):
            M = self._balanced_mask_counts(B_, N, min_m, device)
            K = N - M
            k = torch.ceil(K.float().sqrt()).long().clamp(max=min(H, W)).view(B_, 1)
            top = torch.randint(0, H, (B_, 1), device=device)
            left = torch.randint(0, W, (B_, 1), device=device)

            pos_h = repeat(torch.arange(H, device=device), 'h -> b h w', b=B_, w=W)
            pos_w = repeat(torch.arange(W, device=device), 'w -> b h w', b=B_, h=H)
            pos_h = rearrange(pos_h, 'b h w -> b (h w)')
            pos_w = rearrange(pos_w, 'b h w -> b (h w)')

            dh = (pos_h - top) % H
            dw = (pos_w - left) % W
            in_square = (dh < k) & (dw < k)
            rank = torch.rand(B_, N, device=device).masked_fill(~in_square, -1).argsort(dim=1, descending=True).argsort(dim=1)
            is_visible = rank < K.view(B_, 1)
            is_adjacent = ((dh < k + nonadjacent) | (dh >= H - nonadjacent)) & ((dw < k + nonadjacent) | (dw >= W - nonadjacent))
            is_adjacent &= K.view(B_, 1) > 0
            return ~is_visible, is_adjacent

        is_mask_encoder, is_adjacent_encoder = square_mask(R*B_IN, self.mask_m)
        is_mask_background, is_adjacent_background = square_mask(B_IN, N - min(self.d_max, N))

        is_adjacent_any = (rearrange(is_adjacent_encoder, '(R B) N -> R B N', R=R) | repeat(is_adjacent_background, 'B N -> R B N', R=R)).sum(dim=0) > 2
        is_valid_dp = ~is_adjacent_any
        is_dp_idx = (torch.rand(B_IN, N, device=device) + is_valid_dp.float()).argsort(dim=1, descending=True)[:, :P]
        is_dp = torch.zeros(P*B_IN, N, device=device, dtype=bool).scatter(1, rearrange(is_dp_idx, 'B P -> (P B) 1'), True)
        is_dp = rearrange(is_dp, '(P B) N -> P B N', P=P)

        is_mask_background_dp = repeat(is_mask_background, 'B N -> P B N', P=P) & (~is_dp)
        d_max = min(self.d_max, N)
        is_selected_background_dp = torch.zeros(P, B_IN, N, dtype=bool, device=device).scatter(2, is_mask_background_dp.int().argsort(2)[:,:,:d_max+1], True)
        is_selected_background_dp = rearrange(is_selected_background_dp, 'P B N -> (P B) N')
        is_mask_background_dp = rearrange(is_mask_background_dp, 'P B N -> (P B) N')

        is_valid_target = torch.cat([~is_adjacent_background, ~is_adjacent_encoder], dim=0)

        def get_selected(mask, num_selected):
            return torch.zeros_like(mask).scatter(1, torch.rand_like(mask.float()).masked_fill(~mask, -1).argsort(dim=1)[:, : num_selected], True)
        
        num_selected_addp = max(N-self.mask_m, self.d_max+1)
        is_mask_A = is_mask_encoder
        is_selected_A = get_selected(is_mask_encoder, num_selected_addp)
        is_mask_D = is_mask_background
        is_selected_D = get_selected(is_mask_background, num_selected_addp)
        is_mask_Dp = is_mask_background_dp
        is_selected_Dp = get_selected(is_mask_background_dp, num_selected_addp)
        is_mask_AD = is_mask_A & repeat(is_mask_D, 'B N -> (R B) N', R=R)
        is_selected_AD = get_selected(is_mask_AD, N - self.mask_m + self.d_max)
        is_patch_target = is_dp

        return is_selected_A, is_mask_A, is_selected_D, is_mask_D, is_selected_Dp, is_mask_Dp, is_selected_AD, is_mask_AD, is_patch_target, is_valid_target

    def sample_m_reprv2_1_block(self, B_IN, N, device, H=16, W=16, nonadjacent=0, R=1, is_valid_target=None, **kwargs):
        # Guarantees 
        # m_index 1 has min_mask_ratio selected
        # m_index 2 ~ U[min_mask_ratio,1] per batch index masked. 
        #   Global sum equal for compile
        # is_mask_background_2: background without D, shape b,n
        # is_mask_background_2_dp: background without DP, exactly 1 fewer true per element, shape d,b,n
        # is_mask_background_1_dp: background without DP MAE encoder mask  TODO clever 2x speedup.
        # GIGA TODO: nonadjacent
        
        def block_mask(B, N, M, block=4):
            # Slop code that works
            assert H * W == N and 0 <= M <= N and block >= 1

            self.mask_m

            # balanced counts over [M, N], exact total B * (N + M) // 2
            L = N - M + 1
            q, r = divmod(B, L)
            levels = M + torch.arange(L, device=device)
            counts = levels.repeat(q)

            if r:
                target = B * (N + M) // 2
                rem = target - q * levels.sum()
                base = torch.arange(r, device=device)
                extra = rem - (r * M + r * (r - 1) // 2)
                counts = torch.cat([
                    counts,
                    M + base + extra // r + (base >= r - extra % r).long()
                ])

            counts = counts[torch.randperm(B, device=device)]

            # blocky random priority field
            p0, p1 = (block - 1) // 2, block // 2
            x = torch.rand(B, 1, H, W, device=device)
            scores = F.max_pool2d(
                F.pad(x, (p0, p1, p0, p1), mode="circular"),
                kernel_size=block,
                stride=1,
            ).flatten(1)

            scores = scores + 1e-6 * torch.rand_like(scores)
            rank = scores.argsort(dim=1, descending=True).argsort(dim=1)

            return rank < counts[:, None]

        P = self.d_num_p

        is_mask_encoder = block_mask(R*B_IN,N,self.mask_m)

        is_valid_mask = rearrange(is_mask_encoder, 'RB (H W) -> RB H W', H=H, W=W).clone()
        is_valid_target = is_valid_mask.clone() if is_valid_target is None else rearrange(is_valid_target(is_mask_encoder), 'RB (H W) -> RB H W', H=H, W=W)
        if nonadjacent == 1:
            is_valid_target[:,:-1,:]    &= is_valid_mask[:,1:,:]
            is_valid_target[:,:,:-1]    &= is_valid_mask[:,:,1:]
            is_valid_target[:,1:,:]     &= is_valid_mask[:,:-1,:]
            is_valid_target[:,:,1:]     &= is_valid_mask[:,:,:-1]
            is_valid_target[:,:-1,:-1]  &= is_valid_mask[:,1:,1:]
            is_valid_target[:,:-1,1:]   &= is_valid_mask[:,1:,:-1]
            is_valid_target[:,1:,:-1]   &= is_valid_mask[:,:-1,1:]
            is_valid_target[:,1:,1:]    &= is_valid_mask[:,:-1,:-1]
        # is_valid_target[:,:-2,:]    &= is_valid_mask[:,2:,:]
        # is_valid_target[:,:,:-2]    &= is_valid_mask[:,:,2:]
        # is_valid_target[:,2:,:]     &= is_valid_mask[:,:-2,:]
        # is_valid_target[:,:,2:]     &= is_valid_mask[:,:,:-2]
        is_valid_target = rearrange(is_valid_target, 'RB H W -> RB (H W)')

        is_dp_idx = (repeat(~is_valid_target, '(R B) N -> R B N', R=R).sum(dim=0) + torch.rand((B_IN,N), device=device) * 0.5).argsort(dim=1)[:,:P]
        is_dp = torch.zeros(P*B_IN, N, device=device, dtype=bool).scatter(1, rearrange(is_dp_idx, 'b p -> (p b) 1'), 1)
        is_dp = rearrange(is_dp, '(P B) N -> P B N', P=P)

        d_max = min(self.d_max, N)
        k = torch.randint(0, d_max + 1, (B_IN, 1), device=device)  # how many False per row
        is_background_idx = (torch.rand(B_IN, N, device=device) + is_dp.any(0)).argsort(dim=1)[:, :d_max]  # 0% chance take is_dp
        is_mask_background = torch.ones(B_IN, N, dtype=torch.bool, device=device).scatter(1, is_background_idx, torch.arange(d_max, device=device).unsqueeze(0) >= k)
        is_mask_background_dp = repeat(is_mask_background, 'B N -> P B N', P=P) & (~is_dp)
        is_selected_background_dp = torch.zeros(P, B_IN, N, dtype=bool, device=device).scatter(2, is_mask_background_dp.int().argsort(2)[:,:,:d_max+1], True)
        
        is_selected_background = is_selected_background_dp[0]  # Any element works
        is_selected_background_dp = rearrange(is_selected_background_dp, 'P B N -> (P B) N')
        is_mask_background_dp = rearrange(is_mask_background_dp, 'P B N -> (P B) N')

        is_mask = torch.cat([is_mask_background,is_mask_encoder],dim=0)  # GIGA TODO REQUIRES IS D < IS M
        is_selected_encoder = torch.zeros_like(is_mask).scatter(1, torch.rand_like(is_mask.float()).masked_fill(~is_mask, -1).argsort(dim=1)[:, : N - self.mask_m], True)
        is_valid_target = torch.cat([is_mask_background, is_valid_target], dim=0)


        def get_selected(mask, num_selected):
            return torch.zeros_like(mask).scatter(1, torch.rand_like(mask.float()).masked_fill(~mask, -1).argsort(dim=1)[:, : num_selected], True)
        
        num_selected_addp = max(N-self.mask_m, self.d_max+1)
        is_mask_A = is_mask_encoder
        is_selected_A = get_selected(is_mask_encoder, num_selected_addp)
        is_mask_D = is_mask_background
        is_selected_D = get_selected(is_mask_background, num_selected_addp)
        is_mask_Dp = is_mask_background_dp
        is_selected_Dp = get_selected(is_mask_background_dp, num_selected_addp)
        is_mask_AD = is_mask_A & repeat(is_mask_D, 'B N -> (R B) N', R=R)
        is_selected_AD = get_selected(is_mask_AD, N - self.mask_m + self.d_max)
        is_patch_target = is_dp

        return is_selected_A, is_mask_A, is_selected_D, is_mask_D, is_selected_Dp, is_mask_Dp, is_selected_AD, is_mask_AD, is_patch_target, is_valid_target

    def get_mask(self, strategy, B_IN, N, device, R=1, **kwargs):
        if strategy == 'random':
            return self.sample_m_reprv2_1_random(B_IN, N, device, R=R, **kwargs)
        if strategy == 'block':
            return self.sample_m_reprv2_1_block(B_IN, N, device, R=R, **kwargs)
        if strategy == 'square':
            return self.sample_m_reprv2_1_square(B_IN, N, device, R=R, **kwargs)
        raise ValueError(f'Unknown mask strategy: {strategy}')

    @torch.compile
    def forward_mar_only(self, x, x_target=None, labels=None):
        # Merging the first iteration with different targets
        assert self.in_context_len > 0
        assert self.source_time_repeat == 1

        B_IN, C, H, W = x.shape
        device = x.device
        x_p_og = self.patchify(x)
        z_t1_og_nopos = self.x_embedder(x_p_og)
        dtype = z_t1_og_nopos.dtype

        if x_target is None:
            x_p_target = x_p_og
        else:
            x_p_target = x_target

        N = self.num_patches
        D = self.hidden_size

        P=self.d_num_p
        R=self.batch_repeat_mar
        O=self.diff_repeat
        S=self.source_time_repeat
        U=self.repr_noise_bins
        V=self.diff_bin_repeat_repr

        is_selected_A, is_mask_A, is_selected_D, is_mask_D, is_selected_Dp, is_mask_Dp, is_selected_AD, is_mask_AD, is_patch_target, is_valid_target = self.get_mask(self.mask_strategy_mar, B_IN, N, device, R=R)
        is_mask_encoder_AD = rearrange(is_mask_AD[is_selected_AD], '(X m) -> X m', X=R*B_IN)

        def get_factor_pos(is_mask):
            device = is_mask.device
            B_ = is_mask.size(0)
            is_mask_rc = rearrange(is_mask, 'b (h w) -> b h w', h=self.hw_len)
            pos_h = repeat(torch.arange(self.hw_len, device=device), 'h -> b h w', b=B_, w=self.hw_len)
            pos_w = repeat(torch.arange(self.hw_len, device=device), 'w -> b h w', b=B_, h=self.hw_len)
            mean_pos_h = ((pos_h*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            mean_pos_w = ((pos_w*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            offset_pos_h = rearrange(pos_h, 'b h w -> b (h w)') - mean_pos_h.unsqueeze(-1)
            offset_pos_w = rearrange(pos_w, 'b h w -> b (h w)') - mean_pos_w.unsqueeze(-1)
            offset_idx = (offset_pos_h + self.hw_len - 1 + self.factor_pad) * (2 * self.hw_len - 1 + 2 * self.factor_pad) + (offset_pos_w + self.hw_len - 1 + self.factor_pad)
            # factor_pos = repeat(self.factor_posembed.to(dtype), 'f d -> b f d', b=B_).gather(1, repeat(offset_idx, 'b n -> b n d', d=self.hidden_size))
            factor_pos = F.embedding(offset_idx, self.factor_posembed.to(dtype))
            
            return factor_pos
        
        z_AD_factor_pos = get_factor_pos(is_mask_AD)

        z_t1_A_nopos = repeat(z_t1_og_nopos, 'B N D -> (R B) N D', R=R)
        if self.noise_a:
            x_p_A = repeat(x_p_og, 'B N D -> (R B) N D', R=R)
            noise_level_A = torch.rand(R*B_IN, 1, 1, device=device)
            z_t1_A_nopos = self.x_embedder((1-noise_level_A) * x_p_A + noise_level_A * torch.randn_like(x_p_A))

        z_t1_AD = torch.where(
            repeat(is_mask_D, 'B N -> (R B) N 1', R=R),
            z_t1_A_nopos,
            repeat(z_t1_og_nopos, 'B N D -> (R B) N D', R=R),
        ) + z_AD_factor_pos
                
        def get_encoded(z, is_selected, is_mask_encoder):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, D)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            if self.y_embedder is not None:
                z_e[:, :self.in_context_len] += self.y_embedder(torch.where(torch.rand(B_, device=device) < 0.1, self.num_classes, labels[:B_IN].repeat(R)))[:, None].to(dtype)
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic, "separate")
            for block_idx, block in enumerate(self.encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
        
        z_e_AD_t1 = get_encoded(z_t1_AD, is_selected_AD, is_mask_encoder_AD)

        z_d_AD_t1_nomaskpos = z_e_AD_t1.new_zeros((R*B_IN,N,D)).masked_scatter(
            repeat(is_selected_AD, 'B N -> B N D', D=D),
            z_e_AD_t1[:, self.in_context_len:].flatten(),
        )
        z_d = torch.cat([
            z_e_AD_t1[:, :self.in_context_len],
            torch.where(
                repeat(is_mask_AD, 'RB N -> RB N 1'),
                repeat(self.mask_token.to(dtype), '1 1 D -> (R B) N D', R=R, B=B_IN, N=N),
                z_d_AD_t1_nomaskpos
            )
        ], dim=1)
        z_d = z_d + torch.cat([z_AD_factor_pos.new_zeros(R*B_IN, self.in_context_len, D), z_AD_factor_pos], dim=1)
        for block_idx, block in enumerate(self.decoder):
            z_d = block(z_d, self.feat_rope_incontext, attn_mask=None)

        # Section: Trains such that decoder used to merging multiple encoder inputs.
        # Tradeoff: This wastes compute by calculating D tokens during A mask. But necessary for constant masks
        # Masking is coupled here, but correct because ultimately is_mask_D valid check and is_mask_A masking combine to work
        z_m = repeat(z_d[:,self.in_context_len:][is_mask_A], 'Y D -> (O Y) D', O=O)
        feats_p_rsb = repeat(x_p_target, 'B N D -> (R S B) N D', R=R, S=S)
        feats_p_m = repeat(feats_p_rsb[is_mask_A], 'Y D -> (O Y) D', O=O, D=feats_p_rsb.size(-1))
        t_m = torch.where(torch.rand(z_m.size(0), 1, device=device) < 0.1, torch.rand(z_m.size(0), 1, device=device), torch.sigmoid(torch.randn(z_m.size(0), 1, device=device) * self.std_mar + self.mu_mar))
        eps_m = torch.randn_like(feats_p_m)
        feats_p_m_t = t_m * feats_p_m + (1-t_m) * eps_m
        feats_pred = self.diff_mlp(feats_p_m_t, t_m.flatten(), z_m)
        mse_is_valid = repeat(repeat(is_mask_D, 'B N -> (R S B) N', R=R, S=S)[is_mask_A], 'Y -> (O Y) 1', O=O)
        mse_loss = 2*(F.huber_loss(feats_pred*mse_is_valid, feats_p_m*mse_is_valid, reduction='none', delta=10) / (1-t_m).clamp_min(0.33).pow(2)).mean() / mse_is_valid.float().mean()
        
        loss = mse_loss
        return loss

    @torch.compile
    def forward_reprv2_4_repr_jointenc(self, x, x_target=None, labels=None):
        # Merging the first iteration with different targets
        # Goal: include DP repeat
        assert self.in_context_len > 0
        assert self.source_time_repeat == 1
        assert self.repr_noise_bins == 1


        B_IN, C, H, W = x.shape
        device = x.device
        x_p_og = self.patchify(x)
        z_t1_og_nopos = self.x_embedder(x_p_og)
        dtype = z_t1_og_nopos.dtype

        if x_target is None:
            x_p_target = x_p_og
            z_target = z_t1_og_nopos
        else:
            x_p_target = x_target
            z_target = self.feats_embedder(x_target)

        N = self.num_patches
        D = self.hidden_size

        P=self.d_num_p
        R=self.batch_repeat
        R1=self.batch_repeat+1
        O=self.diff_repeat
        S=self.source_time_repeat
        U=self.repr_noise_bins
        V=self.diff_bin_repeat_repr

        is_selected_A, is_mask_A, is_selected_D, is_mask_D, is_selected_Dp, is_mask_Dp, is_selected_AD, is_mask_AD, is_patch_target, is_valid_target = self.get_mask(self.mask_strategy_repr, B_IN, N, device, R=R)
        is_selected_DADp = torch.cat([is_selected_D, is_selected_A, is_selected_Dp], 0)
        is_mask_DADp = torch.cat([is_mask_D, is_mask_A, is_mask_Dp], 0)
        is_mask_encoder_DADp = rearrange(is_mask_DADp[is_selected_DADp], '(X m) -> X m', X=(1+R+P)*B_IN)
        is_mask_encoder_AD = rearrange(is_mask_AD[is_selected_AD], '(X m) -> X m', X=R*B_IN)

        def get_factor_pos(is_mask):
            device = is_mask.device
            B_ = is_mask.size(0)
            is_mask_rc = rearrange(is_mask, 'b (h w) -> b h w', h=self.hw_len)
            pos_h = repeat(torch.arange(self.hw_len, device=device), 'h -> b h w', b=B_, w=self.hw_len)
            pos_w = repeat(torch.arange(self.hw_len, device=device), 'w -> b h w', b=B_, h=self.hw_len)
            mean_pos_h = ((pos_h*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            mean_pos_w = ((pos_w*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            offset_pos_h = rearrange(pos_h, 'b h w -> b (h w)') - mean_pos_h.unsqueeze(-1)
            offset_pos_w = rearrange(pos_w, 'b h w -> b (h w)') - mean_pos_w.unsqueeze(-1)
            offset_idx = (offset_pos_h + self.hw_len - 1 + self.factor_pad) * (2 * self.hw_len - 1 + 2 * self.factor_pad) + (offset_pos_w + self.hw_len - 1 + self.factor_pad)
            # factor_pos = repeat(self.factor_posembed.to(dtype), 'f d -> b f d', b=B_).gather(1, repeat(offset_idx, 'b n -> b n d', d=self.hidden_size))
            factor_pos = F.embedding(offset_idx, self.factor_posembed.to(dtype))
            
            pad_pos_h = repeat(torch.arange(self.pad_hw, device=device), 'h -> b h w', b=B_, w=self.pad_hw)
            pad_pos_w = repeat(torch.arange(self.pad_hw, device=device), 'w -> b h w', b=B_, h=self.pad_hw)
            pad_offset_idx = (pad_pos_h - mean_pos_h[:,None,None] + self.hw_len - 1) * (self.pad_hw + self.hw_len - 1) + (pad_pos_w - mean_pos_w[:,None,None] + self.hw_len - 1)
            # factor_pos_pad = repeat(self.factor_posembed.to(dtype), 'f d -> b f d', b=B_).gather(1, repeat(pad_offset_idx, 'b h w -> b (h w) d', d=self.hidden_size))
            factor_pos_pad = F.embedding(rearrange(pad_offset_idx, 'b h w -> b (h w)'), self.factor_posembed.to(dtype))

            patch_rope_idx = rearrange((pos_h + self.factor_pad) * self.pad_hw + (pos_w + self.factor_pad), 'b h w -> b (h w)')
            pad_rope_idx = repeat(torch.arange(self.pad_hw * self.pad_hw, device=device), 'q -> b q', b=B_)
            cls_rope_idx = (mean_pos_h + self.factor_pad) * self.pad_hw + mean_pos_w + self.factor_pad

            return factor_pos, factor_pos_pad, patch_rope_idx, pad_rope_idx, cls_rope_idx
        
        z_DADp_factor_pos, z_DADp_factor_pos_pad, z_DADp_encoder_rope_idx, z_DADp_factorpad_rope_idx, z_DADp_cls_rope_idx = get_factor_pos(is_mask_DADp)
        z_AD_factor_pos, z_AD_factor_pos_pad, z_AD_encoder_rope_idx, z_AD_factorpad_rope_idx, z_AD_cls_rope_idx = get_factor_pos(is_mask_AD)

        z_t1_A_nopos = repeat(z_t1_og_nopos, 'B N D -> (R B) N D', R=R)
        if self.noise_a:
            x_p_A = repeat(x_p_og, 'B N D -> (R B) N D', R=R)
            noise_level_A = torch.rand(R*B_IN, 1, 1, device=device)
            z_t1_A_nopos = self.x_embedder((1-noise_level_A) * x_p_A + noise_level_A * torch.randn_like(x_p_A))

        z_t1_DADp = torch.cat([
            z_t1_og_nopos,
            z_t1_A_nopos,
            torch.where(
                rearrange(is_patch_target, 'P B N -> (P B) N 1'), 
                repeat(z_target, "B N D -> (P B) N D", P=P) + rearrange(self.is_d_embed.to(dtype), '1 D -> 1 1 D'), 
                repeat(z_t1_og_nopos, 'B N D -> (P B) N D', P=P),
            )
        ], dim=0) + z_DADp_factor_pos
        z_t1_AD = torch.where(
            repeat(is_mask_D, 'B N -> (R B) N 1', R=R),
            z_t1_A_nopos,
            repeat(z_t1_og_nopos, 'B N D -> (R B) N D', R=R),
        ) + z_AD_factor_pos
                
        labels_DADp = labels_AD = None
        if self.y_embedder is not None:
            drop_D, drop_A = torch.rand((1+R)*B_IN, device=device).lt(0.1).split([B_IN, R*B_IN])
            labels_DADp = torch.where(torch.cat([drop_D, drop_A, drop_D.repeat(P)]), self.num_classes, labels[:B_IN].repeat(1+R+P))
            labels_AD = torch.where(drop_A & drop_D.repeat(R), self.num_classes, labels[:B_IN].repeat(R))

        def get_encoded(z, is_selected, is_mask_encoder, view_labels):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, D)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            if self.y_embedder is not None:
                z_e[:, :self.in_context_len] += self.y_embedder(view_labels)[:, None].to(dtype)
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic, "separate")
            for block_idx, block in enumerate(self.encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
        
        z_e_DADp_t1 = get_encoded(z_t1_DADp, is_selected_DADp, is_mask_encoder_DADp, labels_DADp)
        z_e_AD_t1 = get_encoded(z_t1_AD, is_selected_AD, is_mask_encoder_AD, labels_AD)

        is_mask_DAD = torch.cat([is_mask_D, is_mask_AD], dim=0)
        z_d_D_t1_nomaskpos = z_e_DADp_t1.new_zeros((B_IN,N,D)).masked_scatter(
            repeat(is_selected_D, 'B N -> B N D', D=D),
            z_e_DADp_t1[:B_IN, self.in_context_len:].flatten(),
        )
        z_d_AD_t1_nomaskpos = z_e_AD_t1.new_zeros((R*B_IN,N,D)).masked_scatter(
            repeat(is_selected_AD, 'B N -> B N D', D=D),
            z_e_AD_t1[:, self.in_context_len:].flatten(),
        )
        z_d_DAD_t1_nomaskpos = torch.cat([z_d_D_t1_nomaskpos, z_d_AD_t1_nomaskpos], dim=0)
        z_d = torch.cat([
            torch.cat([z_e_DADp_t1[:B_IN, :self.in_context_len], z_e_AD_t1[:, :self.in_context_len]], dim=0),
            torch.where(
                repeat(is_mask_DAD, 'RB N -> RB N 1'),
                repeat(self.mask_token.to(dtype), '1 1 D -> (R B) N D', R=1+R, B=B_IN, N=N),
                z_d_DAD_t1_nomaskpos
            )
        ], dim=1)
        z_DAD_factor_pos = torch.cat([z_DADp_factor_pos[:B_IN], z_AD_factor_pos], dim=0)
        z_d = z_d + torch.cat([z_DAD_factor_pos.new_zeros((1+R)*B_IN, self.in_context_len, D), z_DAD_factor_pos], dim=1)
        for block_idx, block in enumerate(self.decoder):
            z_d = block(z_d, self.feat_rope_incontext, attn_mask=None)

        # Section: Trains such that decoder used to merging multiple encoder inputs.
        # Tradeoff: This wastes compute by calculating D tokens during A mask. But necessary for constant masks
        # Masking is coupled here, but correct because ultimately is_mask_D valid check and is_mask_A masking combine to work
        is_mask_DA = torch.cat([is_mask_D, is_mask_A], dim=0)
        z_m = repeat(z_d[:,self.in_context_len:][is_mask_DA], 'Y D -> (O Y) D', O=O)
        feats_p_rsb = repeat(x_p_target, 'B N D -> (R S B) N D', R=R1, S=S)
        feats_p_m = repeat(feats_p_rsb[is_mask_DA], 'Y D -> (O Y) D', O=O, D=feats_p_rsb.size(-1))
        t_m = torch.where(torch.rand(z_m.size(0), 1, device=device) < 0.1, torch.rand(z_m.size(0), 1, device=device), torch.sigmoid(torch.randn(z_m.size(0), 1, device=device) * self.std_mar + self.mu_mar))
        eps_m = torch.randn_like(feats_p_m)
        feats_p_m_t = t_m * feats_p_m + (1-t_m) * eps_m
        feats_pred = self.diff_mlp(feats_p_m_t, t_m.flatten(), z_m)
        mse_is_valid = repeat(repeat(is_mask_D, 'B N -> (R S B) N', R=R1, S=S)[is_mask_DA], 'Y -> (O Y) 1', O=O)
        mse_loss = 2*(F.huber_loss(feats_pred*mse_is_valid, feats_p_m*mse_is_valid, reduction='none', delta=10) / (1-t_m).clamp_min(0.33).pow(2)).mean() / mse_is_valid.float().mean()
        
        with torch.no_grad():
            z_d_rs = rearrange(z_d[:,self.in_context_len:], '(R S B) N D -> R S B N D', R=R1, S=S)
            z_dp_rsp = z_d_rs[:, :, torch.arange(B_IN, device=device), is_patch_target.int().argmax(-1)].float()
            z_dp_uvprs = repeat(z_dp_rsp, 'R S P B D -> (U V P R S B) D', U=U, V=V)

            x_dp_uvprs_t1 = repeat(repeat(x_p_target, 'B N D -> P B N D', P=P)[is_patch_target], '(P B) D -> (U V P R S B) D', P=P, R=R1, S=S, U=U, V=V).float()
            tu_dp_uvprs = (torch.rand(U,V,P,S,B_IN, device=device) if self.is_raw_pmi else torch.sigmoid(torch.randn(U,V,P,S,B_IN, device=device) * self.std_repr + self.mu_repr)) * (self.noise_bin_limits[1:]-self.noise_bin_limits[:-1]).float().view(-1,1,1,1,1) + self.noise_bin_limits[:-1].view(-1,1,1,1,1)
            tu_dp_uvprs = repeat(tu_dp_uvprs, 'U V P S B -> (U V P R S B) 1', R=R1).float()
            eps_dp_uvprs_tu = repeat(torch.randn(U,V,P,S,B_IN,x_dp_uvprs_t1.size(-1),device=device), 'U V P S B D -> (U V P R S B) D', R=R1).float()
            x_dp_uvprs_tu = tu_dp_uvprs * x_dp_uvprs_t1 + (1-tu_dp_uvprs) * eps_dp_uvprs_tu

            x_dp_qprs_pred = self.diff_mlp(x_dp_uvprs_tu.to(dtype), tu_dp_uvprs.flatten().to(dtype), z_dp_uvprs.to(dtype)).float()

            freq_weight = 1/2*tu_dp_uvprs/(1-tu_dp_uvprs).clamp_min(0.05).pow(3) if self.is_raw_pmi else 1/2*tu_dp_uvprs/(1-tu_dp_uvprs).clamp_min(0.33).pow(2)
            h_p_uvprb_pred_only = freq_weight * (x_dp_qprs_pred - x_dp_uvprs_t1).pow(2).sum(-1, True)
            h_p_rbpuv_pred_only = rearrange(h_p_uvprb_pred_only, '(U V P R B) 1 -> R B P U V', P=P, R=R1, U=U, V=V).to(dtype)

        # REPR Calculation
        def get_repr_encoder(z_e, is_selected, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic, "separate")
            for block_idx, block in enumerate(self.repr_encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
            
        def get_repr_decoder(z_e, repr_factor_pos_pad, factorpad_rope_idx, encoder_rope_idx, cls_rope_idx, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)

            attn_mask = rearrange(~is_mask_encoder_ic, 'B K -> B 1 1 K')
            
            repr_tokens = repr_factor_pos_pad
            q_rope = partial(self.feat_rope_incontext_pad, idx=factorpad_rope_idx, cls_idx=None, use_cls=False)
            k_rope = partial(self.feat_rope_incontext_pad, idx=encoder_rope_idx, cls_idx=cls_rope_idx, use_cls=True)
            for block in self.representation_encoder:
                repr_tokens = block(repr_tokens, z_e, q_rope, k_rope, attn_mask=attn_mask)
            return repr_tokens

        if self.is_repr_encoderdecoder:
            z_re_DADp_t1 = get_repr_encoder(z_e_DADp_t1, is_selected_DADp, is_mask_encoder_DADp)
        else:
            z_re_DADp_t1 = z_e_DADp_t1

        z_DADp_t1_repr = get_repr_decoder(z_re_DADp_t1, z_DADp_factor_pos_pad, z_DADp_factorpad_rope_idx, z_DADp_encoder_rope_idx[is_selected_DADp].view((1+R+P)*B_IN,-1), z_DADp_cls_rope_idx, is_mask_encoder_DADp)
        z_DA_t1_repr = rearrange(self.repr_final_layer(z_DADp_t1_repr[:(1+R)*B_IN]), '(R b) q d -> R b 1 q d', R=R1)
        z_Dp_t1_repr = repeat(self.repr_final_layer(z_DADp_t1_repr[(1+R)*B_IN:]), '(P b) q (U d) -> b P U q d', P=P, U=U)
        z_t1_repr_global = rearrange(self.repr_final_layer_global(z_re_DADp_t1[:(1+R)*B_IN, :self.in_context_len]), '(R b) C d -> R b 1 1 (C d)', R=R1)
        z_Dp_t1_repr_global = repeat(self.repr_final_layer_global(z_re_DADp_t1[(1+R)*B_IN:, :self.in_context_len]), '(P b) C d -> b P U (C d)', P=P, U=U)

        z_DA_t1_repr = torch.where(torch.rand(z_DA_t1_repr.size(0),device=device).view(-1,1,1,1,1) < self.pad_dropout, z_DA_t1_repr*self.pad_dropout_mask.to(dtype), z_DA_t1_repr)

        # MINI TODO REMOVE IS EMPTY
        is_empty_DA = is_mask_DA.all(-1) & (labels_DADp[:R1*B_IN] == self.num_classes if self.y_embedder is not None else True)
        z_DA_t1_repr = torch.where(is_empty_DA.view(R1, B_IN, 1, 1, 1), 0, z_DA_t1_repr)
        z_t1_repr_global = torch.where(is_empty_DA.view(R1, B_IN, 1, 1, 1), 0, z_t1_repr_global)
        z_DA_t1_repr = repeat(z_DA_t1_repr, 'R b P q D -> R b P U q D', U=U) # R b P U q d
        z_D_t1_repr = z_DA_t1_repr[:1]
        z_A_t1_repr = z_DA_t1_repr[1:]
        z_D_t1_repr_global = z_t1_repr_global[:1]
        z_A_t1_repr_global = z_t1_repr_global[1:]

        is_valid_pmi = reduce(~(repeat(~is_valid_target, '(R b) n -> R b P n', R=R1, P=P) & repeat(is_patch_target, 'P b n -> R b P n', R=R1)), 'R b P n -> R b P 1', 'all')  # don't allow loss for: ~is_mask and is_dp
        is_valid_D = is_valid_pmi[:1]
        is_valid_A = is_valid_pmi[1:]
        is_valid = is_valid_D & is_valid_A

        h_p_given_D_preintegration = h_p_rbpuv_pred_only[:1]
        h_p_given_DA_preintegration = h_p_rbpuv_pred_only[1:]

        pmi_pred = torch.einsum('rbuqd,bpuqd->rbpu', z_A_t1_repr[:,:,0], z_Dp_t1_repr - z_D_t1_repr[0]) / (2 * z_A_t1_repr.size(-1) * z_A_t1_repr.size(-2)**0.5) * D**0.5  # GIGA TODO USE REPR HERE!
        pmi_pred_global = torch.einsum('rbud,bpud->rbpu', z_A_t1_repr_global[:,:,0], z_Dp_t1_repr_global - z_D_t1_repr_global[0]) / (2 * z_A_t1_repr_global.size(-1)**0.5)
        pmi_target = (h_p_given_D_preintegration - h_p_given_DA_preintegration).mean(-1) * self.pmi_target_mul  # GIGA TODO THIS CLAMPING IS ONLY GOOD FOR PIXELS (BUT SANCHECKED)

        # Factorization has an optimum at zero, so huber should be fine, no? Only kind of due to E[PMI est error], but PMI est bias dominates regardless
        repr_loss = (2*F.huber_loss(pmi_pred, pmi_target, reduction='none', delta=self.repr_delta) * is_valid / reduce(is_valid.bfloat16(), 'R b P U -> R b 1 1', 'mean').clamp_min(1e-5)).mean()
        repr_loss_global = (2*F.huber_loss(pmi_pred_global, pmi_target, reduction='none', delta=self.repr_delta) * is_valid / reduce(is_valid.bfloat16(), 'R b P U -> R b 1 1', 'mean').clamp_min(1e-5)).mean()

        loss = mse_loss + self.repr_loss_weight * repr_loss + self.repr_loss_weight * repr_loss_global
        return loss, mse_loss.detach().clone(), repr_loss.detach().clone(), repr_loss_global.detach().clone()


    @torch.compile
    def forward_reprv2_4_repr_jointenc_frozenmar(self, x, frozen_self, x_target=None):
        # Merging the first iteration with different targets
        # Goal: include DP repeat
        assert self.in_context_len > 0
        assert self.source_time_repeat == 1
        assert self.repr_noise_bins == 1


        B_IN, C, H, W = x.shape
        device = x.device
        x_p_og = self.patchify(x)
        z_t1_og_nopos = self.x_embedder(x_p_og)
        dtype = z_t1_og_nopos.dtype

        if x_target is None:
            x_p_target = x_p_og
            z_target = z_t1_og_nopos
        else:
            x_p_target = x_target
            z_target = self.feats_embedder(x_target)

        N = self.num_patches
        D = self.hidden_size

        P=self.d_num_p
        R=self.batch_repeat
        R1=self.batch_repeat+1
        S=self.source_time_repeat
        U=self.repr_noise_bins
        V=self.diff_bin_repeat_repr

        is_selected_A, is_mask_A, is_selected_D, is_mask_D, is_selected_Dp, is_mask_Dp, is_selected_AD, is_mask_AD, is_patch_target, is_valid_target = self.get_mask(self.mask_strategy_repr, B_IN, N, device, R=R)
        is_selected_DADp = torch.cat([is_selected_D, is_selected_A, is_selected_Dp], 0)
        is_mask_DADp = torch.cat([is_mask_D, is_mask_A, is_mask_Dp], 0)
        is_mask_encoder_DADp = rearrange(is_mask_DADp[is_selected_DADp], '(X m) -> X m', X=(1+R+P)*B_IN)
        is_mask_encoder_AD = rearrange(is_mask_AD[is_selected_AD], '(X m) -> X m', X=R*B_IN)
        is_mask_DAD = torch.cat([is_mask_D, is_mask_AD], dim=0)

        def get_factor_pos(self, is_mask):
            device = is_mask.device
            B_ = is_mask.size(0)
            is_mask_rc = rearrange(is_mask, 'b (h w) -> b h w', h=self.hw_len)
            pos_h = repeat(torch.arange(self.hw_len, device=device), 'h -> b h w', b=B_, w=self.hw_len)
            pos_w = repeat(torch.arange(self.hw_len, device=device), 'w -> b h w', b=B_, h=self.hw_len)
            mean_pos_h = ((pos_h*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            mean_pos_w = ((pos_w*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            offset_pos_h = rearrange(pos_h, 'b h w -> b (h w)') - mean_pos_h.unsqueeze(-1)
            offset_pos_w = rearrange(pos_w, 'b h w -> b (h w)') - mean_pos_w.unsqueeze(-1)
            offset_idx = (offset_pos_h + self.hw_len - 1 + self.factor_pad) * (2 * self.hw_len - 1 + 2 * self.factor_pad) + (offset_pos_w + self.hw_len - 1 + self.factor_pad)
            factor_pos = repeat(self.factor_posembed.to(dtype), 'f d -> b f d', b=B_).gather(1, repeat(offset_idx, 'b n -> b n d', d=self.hidden_size))
            
            pad_pos_h = repeat(torch.arange(self.pad_hw, device=device), 'h -> b h w', b=B_, w=self.pad_hw)
            pad_pos_w = repeat(torch.arange(self.pad_hw, device=device), 'w -> b h w', b=B_, h=self.pad_hw)
            pad_offset_idx = (pad_pos_h - mean_pos_h[:,None,None] + self.hw_len - 1) * (self.pad_hw + self.hw_len - 1) + (pad_pos_w - mean_pos_w[:,None,None] + self.hw_len - 1)
            factor_pos_pad = repeat(self.factor_posembed.to(dtype), 'f d -> b f d', b=B_).gather(1, repeat(pad_offset_idx, 'b h w -> b (h w) d', d=self.hidden_size))

            patch_rope_idx = rearrange((pos_h + self.factor_pad) * self.pad_hw + (pos_w + self.factor_pad), 'b h w -> b (h w)')
            pad_rope_idx = repeat(torch.arange(self.pad_hw * self.pad_hw, device=device), 'q -> b q', b=B_)
            cls_rope_idx = (mean_pos_h + self.factor_pad) * self.pad_hw + mean_pos_w + self.factor_pad

            return factor_pos, factor_pos_pad, patch_rope_idx, pad_rope_idx, cls_rope_idx
        
        z_DADp_factor_pos, z_DADp_factor_pos_pad, z_DADp_encoder_rope_idx, z_DADp_factorpad_rope_idx, z_DADp_cls_rope_idx = get_factor_pos(self, is_mask_DADp)
        z_AD_factor_pos, z_AD_factor_pos_pad, z_AD_encoder_rope_idx, z_AD_factorpad_rope_idx, z_AD_cls_rope_idx = get_factor_pos(self, is_mask_AD)
        z_DAD_factor_pos_frozen, z_DAD_factor_pos_pad_frozen, z_DAD_encoder_rope_idx_frozen, z_DAD_factorpad_rope_idx_frozen, z_DAD_cls_rope_idx_frozen = get_factor_pos(frozen_self, is_mask_DAD)

        t_aug = torch.rand((B_IN, 1, 1), dtype=dtype, device=device) * 0. + 1.
        x_pt_og = t_aug * x_p_og + (1-t_aug) * torch.randn_like(x_p_og)
        z_taug_og_nopos = self.x_embedder(x_pt_og)

        z_t1_DADp = torch.cat([
            z_taug_og_nopos, 
            repeat(z_t1_og_nopos, 'B N D -> (R B) N D', R=R),
            torch.where(
                rearrange(is_patch_target, 'P B N -> (P B) N 1'), 
                repeat(z_target, "B N D -> (P B) N D", P=P) + rearrange(self.is_d_embed.to(dtype), '1 D -> 1 1 D'), 
                repeat(z_taug_og_nopos, 'B N D -> (P B) N D', P=P),
            )
        ], dim=0) + z_DADp_factor_pos
        z_t1_AD = torch.where(
            is_mask_A[...,None],
            repeat(z_taug_og_nopos, 'B N D -> (R B) N D', R=R),
            repeat(z_t1_og_nopos, 'B N D -> (R B) N D', R=R),
        ) + z_AD_factor_pos
                
        def get_encoded(self, z, is_selected, is_mask_encoder):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, z.size(-1))  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(z.dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic, "separate")
            for block_idx, block in enumerate(self.encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
        
        z_e_DADp_t1 = get_encoded(self, z_t1_DADp, is_selected_DADp, is_mask_encoder_DADp)
        with torch.no_grad():
            z_t1_og_nopos_frozen = frozen_self.x_embedder(x_p_og)
            z_taug_og_nopos_frozen = frozen_self.x_embedder(x_pt_og)
            z_t1_D_frozen = z_taug_og_nopos_frozen + z_DAD_factor_pos_frozen[:B_IN]
            z_t1_AD_frozen = torch.where(
                is_mask_A[...,None],
                repeat(z_taug_og_nopos_frozen, 'B N D -> (R B) N D', R=R),
                repeat(z_t1_og_nopos_frozen, 'B N D -> (R B) N D', R=R),
            ) + z_DAD_factor_pos_frozen[B_IN:]

            z_e_D_t1_frozen = get_encoded(frozen_self, z_t1_D_frozen, is_selected_D, is_mask_encoder_DADp[:B_IN])
            z_e_AD_t1_frozen = get_encoded(frozen_self, z_t1_AD_frozen, is_selected_AD, is_mask_encoder_AD)

            z_d_D_t1_nomaskpos = z_e_D_t1_frozen.new_zeros((B_IN,N,D)).masked_scatter(
                repeat(is_selected_D, 'B N -> B N D', D=D),
                z_e_D_t1_frozen[:, frozen_self.in_context_len:].flatten(),
            )
            z_d_AD_t1_nomaskpos = z_e_AD_t1_frozen.new_zeros((R*B_IN,N,D)).masked_scatter(
                repeat(is_selected_AD, 'B N -> B N D', D=D),
                z_e_AD_t1_frozen[:, frozen_self.in_context_len:].flatten(),
            )
            z_d_DAD_t1_nomaskpos = torch.cat([z_d_D_t1_nomaskpos, z_d_AD_t1_nomaskpos], dim=0)
            z_d = torch.cat([
                torch.cat([
                    z_e_D_t1_frozen[:, :frozen_self.in_context_len],
                    z_e_AD_t1_frozen[:, :frozen_self.in_context_len],
                ], dim=0),
                torch.where(
                    repeat(is_mask_DAD, 'RB N -> RB N 1'),
                    repeat(frozen_self.mask_token.to(dtype), '1 1 D -> (R B) N D', R=1+R, B=B_IN, N=N),
                    z_d_DAD_t1_nomaskpos
                )
            ], dim=1)
            z_d = z_d + torch.cat([
                z_DAD_factor_pos_frozen.new_zeros((1+R)*B_IN, frozen_self.in_context_len, D),
                z_DAD_factor_pos_frozen,
            ], dim=1)
            for block_idx, block in enumerate(frozen_self.decoder):
                z_d = block(z_d, frozen_self.feat_rope_incontext, attn_mask=None)

        # No need to train MAR anymore
        is_mask_DA = torch.cat([is_mask_D, is_mask_A], dim=0)
        
        with torch.no_grad():
            z_d_prs = repeat(z_d[:,frozen_self.in_context_len:], '(R S B) N D -> P R S B N D', P=P, R=R1, S=S).float()
            is_dp_prs = repeat(is_patch_target, 'P B N -> P R S B N', R=R1, S=S)
            z_dp_uvprs = repeat(z_d_prs[is_dp_prs], '(X) D -> (U V X) D', U=U, V=V) # (U V P R S B) D

            x_dp_uvprs_t1 = repeat(repeat(x_p_target, 'B N D -> P B N D', P=P)[is_patch_target], '(P B) D -> (U V P R S B) D', P=P, R=R1, S=S, U=U, V=V).float()
            tu_dp_uvprs = torch.sigmoid(torch.randn(U,V,P,S,B_IN, device=device) * frozen_self.std_repr + frozen_self.mu_repr) * (frozen_self.noise_bin_limits[1:]-frozen_self.noise_bin_limits[:-1]).float().view(-1,1,1,1,1) + frozen_self.noise_bin_limits[:-1].view(-1,1,1,1,1)
            tu_dp_uvprs = repeat(tu_dp_uvprs, 'U V P S B -> (U V P R S B) 1', R=R1).float()
            eps_dp_uvprs_tu = repeat(torch.randn(U,V,P,S,B_IN,x_dp_uvprs_t1.size(-1),device=device), 'U V P S B D -> (U V P R S B) D', R=R1).float()
            x_dp_uvprs_tu = tu_dp_uvprs * x_dp_uvprs_t1 + (1-tu_dp_uvprs) * eps_dp_uvprs_tu

            x_dp_qprs_pred = frozen_self.diff_mlp(x_dp_uvprs_tu.to(dtype), tu_dp_uvprs.flatten().to(dtype), z_dp_uvprs.to(dtype)).float()

            freq_weight = 1/2*tu_dp_uvprs/(1-tu_dp_uvprs).clamp_min(0.33).pow(2)  # Now weighted with logit sigmoid
            h_p_uvprb_pred_only = freq_weight * (x_dp_qprs_pred - x_dp_uvprs_t1).pow(2).sum(-1, True)
            h_p_rbpuv_pred_only = rearrange(h_p_uvprb_pred_only, '(U V P R B) 1 -> R B P U V', P=P, R=R1, U=U, V=V).to(dtype)

        # REPR Calculation
        def get_repr_encoder(z_e, is_selected, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic, "separate")
            for block_idx, block in enumerate(self.repr_encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
            
        def get_repr_decoder(z_e, repr_factor_pos_pad, factorpad_rope_idx, encoder_rope_idx, cls_rope_idx, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)

            same_rope = rearrange(factorpad_rope_idx, 'B Q -> B Q 1') == rearrange(encoder_rope_idx, 'B K -> B 1 K')
            same_rope_or_in_context = F.pad(same_rope, (self.in_context_len, 0), value=False)
            q_has_k = reduce(same_rope, 'B Q K -> B Q 1', 'any')
            q_is_restricted = q_has_k & (torch.rand_like(q_has_k, dtype=dtype) < 0.0)
            attn_mask_default = rearrange(~is_mask_encoder_ic, 'B K -> B 1 K')
            attn_mask_restricted = rearrange(~is_mask_encoder_ic, 'B K -> B 1 K') & ((~q_is_restricted) | same_rope_or_in_context)
            attn_mask = repeat(torch.where(q_is_restricted, attn_mask_restricted, attn_mask_default), 'B Q K -> B H Q K', H=self.num_heads)
            
            repr_tokens = repr_factor_pos_pad
            q_rope = partial(self.feat_rope_incontext_pad, idx=factorpad_rope_idx, cls_idx=None, use_cls=False)
            k_rope = partial(self.feat_rope_incontext_pad, idx=encoder_rope_idx, cls_idx=cls_rope_idx, use_cls=True)
            for block in self.representation_encoder:
                repr_tokens = block(repr_tokens, z_e, q_rope, k_rope, attn_mask=attn_mask)
            return repr_tokens

        if self.is_repr_encoderdecoder:
            z_re_DADp_t1 = get_repr_encoder(z_e_DADp_t1, is_selected_DADp, is_mask_encoder_DADp)
        else:
            z_re_DADp_t1 = z_e_DADp_t1

        z_DADp_t1_repr = get_repr_decoder(z_re_DADp_t1, z_DADp_factor_pos_pad, z_DADp_factorpad_rope_idx, z_DADp_encoder_rope_idx[is_selected_DADp].view((1+R+P)*B_IN,-1), z_DADp_cls_rope_idx, is_mask_encoder_DADp)
        z_DA_t1_repr = repeat(self.repr_final_layer(z_DADp_t1_repr[:(1+R)*B_IN]), '(R b) q d -> R b P q d', R=R1, P=P)
        z_Dp_t1_repr = repeat(self.repr_final_layer(z_DADp_t1_repr[(1+R)*B_IN:]), '(P b) q (U d) -> b P U q d', P=P, U=U)
        z_t1_repr_global = repeat(self.repr_final_layer_global(z_re_DADp_t1[:(1+R)*B_IN, :self.in_context_len]), '(R b) C d -> R b P U (C d)', R=R1, P=P, U=U)
        z_Dp_t1_repr_global = repeat(self.repr_final_layer_global(z_re_DADp_t1[(1+R)*B_IN:, :self.in_context_len]), '(P b) C d -> b P U (C d)', P=P, U=U)

        z_DA_t1_repr = torch.where(torch.rand(z_DA_t1_repr.size(0),device=device).view(-1,1,1,1,1) < self.pad_dropout, z_DA_t1_repr*self.pad_dropout_mask.to(dtype), z_DA_t1_repr)

        z_DA_t1_repr = torch.where(reduce(is_mask_DA, '(R b) n -> R b 1 1 1', 'all', R=R1), 0, z_DA_t1_repr) # Special case: zero out (prevents magnitude drift)
        z_t1_repr_global = torch.where(reduce(is_mask_DA, '(R b) n -> R b 1 1 1', 'all', R=R1), 0, z_t1_repr_global)
        z_DA_t1_repr = repeat(z_DA_t1_repr, 'R b P q D -> R b P U q D', U=U) # R b P U q d
        z_D_t1_repr = z_DA_t1_repr[:1]
        z_A_t1_repr = z_DA_t1_repr[1:]
        z_D_t1_repr_global = z_t1_repr_global[:1]
        z_A_t1_repr_global = z_t1_repr_global[1:]

        is_valid_pmi = reduce(~(repeat(~is_valid_target, '(R b) n -> R b P n', R=R1, P=P) & repeat(is_patch_target, 'P b n -> R b P n', R=R1)), 'R b P n -> R b P 1', 'all')  # don't allow loss for: ~is_mask and is_dp
        is_valid_D = is_valid_pmi[:1]
        is_valid_A = is_valid_pmi[1:]
        is_valid = is_valid_D & is_valid_A

        h_p_given_D_preintegration = h_p_rbpuv_pred_only[:1]
        h_p_given_DA_preintegration = h_p_rbpuv_pred_only[1:]

        pmi_pred = reduce(z_A_t1_repr * (z_Dp_t1_repr - z_D_t1_repr) / 2 * z_A_t1_repr.size(-2)**0.5, 'R b P U q d -> R b P U', 'mean') * D**0.5  # GIGA TODO USE REPR HERE!
        pmi_pred_global = reduce(z_A_t1_repr_global * (z_Dp_t1_repr_global - z_D_t1_repr_global) / 2, 'R b P U d -> R b P U', 'mean') * z_A_t1_repr_global.size(-1)**0.5
        pmi_target = (h_p_given_D_preintegration - h_p_given_DA_preintegration).mean(-1) * self.pmi_target_mul  # GIGA TODO THIS CLAMPING IS ONLY GOOD FOR PIXELS (BUT SANCHECKED)

        # Factorization has an optimum at zero, so huber should be fine, no? Only kind of due to E[PMI est error], but PMI est bias dominates regardless
        repr_loss = (2*F.huber_loss(pmi_pred, pmi_target, reduction='none', delta=self.repr_delta) * is_valid / reduce(is_valid.bfloat16(), 'R b P U -> R b 1 1', 'mean').clamp_min(1e-5)).mean()
        repr_loss_global = (2*F.huber_loss(pmi_pred_global, pmi_target, reduction='none', delta=self.repr_delta) * is_valid / reduce(is_valid.bfloat16(), 'R b P U -> R b 1 1', 'mean').clamp_min(1e-5)).mean()

        loss = self.repr_loss_weight * (repr_loss + repr_loss_global)
        return loss, repr_loss.detach().clone(), repr_loss_global.detach().clone()


    def forward_mar_only_crossview(self, x):
        # Merging the first iteration with different targets
        # Goal: include DP repeat
        assert self.in_context_len > 0
        assert self.source_time_repeat == 1

        B_IN, C, H, W = x.shape
        device = x.device

        N = self.num_patches
        D = self.hidden_size

        P=self.d_num_p
        R=1
        O=self.diff_repeat
        S=self.source_time_repeat
        U=self.repr_noise_bins
        V=self.diff_bin_repeat_repr

        def augment_image(x):
            if not self.do_simclr_augs:
                return x, x.new_tensor((0, 0, H, W), dtype=torch.long).repeat(x.size(0), 1)
            x = (x + 1) / 2
            x, random_resize_state = random_resized_crop_batch(x, (H, W))
            x_aug = post_crop_augment_batch(x) * 2 - 1
            return x_aug, random_resize_state

        x_aug_A, x_A_resize_state = augment_image(repeat(x, 'B C H W -> (R B) C H W', R=R))
        x_aug_D, x_D_resize_state = augment_image(x)

        x_p_A = self.patchify(x_aug_A)
        x_p_D = self.patchify(x_aug_D)
        z_A_nopos = self.x_embedder(x_p_A)
        z_D_nopos = self.x_embedder(x_p_D)
        dtype = z_D_nopos.dtype

        is_selected_A, is_mask_A, is_selected_D, is_mask_D, is_selected_Dp, is_mask_Dp, is_selected_AD, is_mask_AD, is_patch_target, is_valid_target = self.get_mask(self.mask_strategy_mar, B_IN, N, device, R=R, check_A_overlap=False)

        is_selected_DADp = torch.cat([is_selected_D, is_selected_A, is_selected_Dp], 0)
        is_mask_DADp = torch.cat([is_mask_D, is_mask_A, is_mask_Dp], 0)

        def get_factor_pos(is_mask):
            device = is_mask.device
            B_ = is_mask.size(0)
            is_mask_rc = rearrange(is_mask, 'b (h w) -> b h w', h=self.hw_len)
            pos_h = repeat(torch.arange(self.hw_len, device=device), 'h -> b h w', b=B_, w=self.hw_len)
            pos_w = repeat(torch.arange(self.hw_len, device=device), 'w -> b h w', b=B_, h=self.hw_len)
            mean_pos_h = 7 + 0 * ((pos_h*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            mean_pos_w = 7 + 0 * ((pos_w*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            offset_pos_h = rearrange(pos_h, 'b h w -> b (h w)') - mean_pos_h.unsqueeze(-1)
            offset_pos_w = rearrange(pos_w, 'b h w -> b (h w)') - mean_pos_w.unsqueeze(-1)
            offset_idx = (offset_pos_h + self.hw_len - 1 + self.factor_pad) * (2 * self.hw_len - 1 + 2 * self.factor_pad) + (offset_pos_w + self.hw_len - 1 + self.factor_pad)
            factor_pos = repeat(self.factor_posembed.to(dtype), 'f d -> b f d', b=B_).gather(1, repeat(offset_idx, 'b n -> b n d', d=self.hidden_size))
            
            pad_pos_h = repeat(torch.arange(self.pad_hw, device=device), 'h -> b h w', b=B_, w=self.pad_hw)
            pad_pos_w = repeat(torch.arange(self.pad_hw, device=device), 'w -> b h w', b=B_, h=self.pad_hw)
            pad_offset_idx = (pad_pos_h - mean_pos_h[:,None,None] + self.hw_len - 1) * (self.pad_hw + self.hw_len - 1) + (pad_pos_w - mean_pos_w[:,None,None] + self.hw_len - 1)
            factor_pos_pad = repeat(self.factor_posembed.to(dtype), 'f d -> b f d', b=B_).gather(1, repeat(pad_offset_idx, 'b h w -> b (h w) d', d=self.hidden_size))

            patch_rope_idx = rearrange((pos_h + self.factor_pad) * self.pad_hw + (pos_w + self.factor_pad), 'b h w -> b (h w)')
            pad_rope_idx = repeat(torch.arange(self.pad_hw * self.pad_hw, device=device), 'q -> b q', b=B_)
            cls_rope_idx = (mean_pos_h + self.factor_pad) * self.pad_hw + mean_pos_w + self.factor_pad

            return factor_pos, factor_pos_pad, patch_rope_idx, pad_rope_idx, cls_rope_idx
        
        z_AD_factor_pos, z_AD_factor_pos_pad, z_AD_encoder_rope_idx, z_AD_factorpad_rope_idx, z_AD_cls_rope_idx = get_factor_pos(is_mask_AD)

        z_t1_AD = torch.cat([
            z_A_nopos + z_AD_factor_pos,
            repeat(z_D_nopos, 'B N D -> (R B) N D', R=R) + z_AD_factor_pos,
        ], dim=1)
        is_selected_AD_cat = torch.cat([is_selected_A, repeat(is_selected_D, 'B N -> (R B) N', R=R)], dim=1)
        is_mask_AD_cat = torch.cat([is_mask_A, repeat(is_mask_D, 'B N -> (R B) N', R=R)], dim=1)
        is_mask_encoder_AD_cat = rearrange(is_mask_AD_cat[is_selected_AD_cat], '(X m) -> X m', X=R*B_IN)
        num_selected_A = is_selected_A[0].sum()
        assert (is_selected_A.sum(1) == num_selected_A).all()
        def get_encoded_crossview(z, is_selected, is_mask_encoder, view_boundary, encoder_rope_idx, cls_rope_idx):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, D)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic, "separate")
            attn_mask_encoder_viewseparate = attn_mask_encoder.clone()
            attn_mask_encoder_viewseparate[:,:,self.in_context_len+view_boundary:,self.in_context_len:self.in_context_len+view_boundary] = False
            attn_mask_encoder_viewseparate[:,:,self.in_context_len:self.in_context_len+view_boundary,self.in_context_len+view_boundary:] = False
            # first todo: rope indices
            # second todo: alternate masking (vggt)
            # giga todo: Factor pad broken
            selected_encoder_rope_idx = torch.cat([encoder_rope_idx, encoder_rope_idx], dim=1)[is_selected].view(B_, -1)
            for block_idx, block in enumerate(self.encoder):
                if block_idx % 2 == 0:
                    z_e = block(z_e, partial(self.feat_rope_incontext_pad, idx=selected_encoder_rope_idx, cls_idx=cls_rope_idx, use_cls=True), attn_mask=attn_mask_encoder_viewseparate)
                else:
                    z_e = block(z_e, partial(self.feat_rope_incontext_pad, idx=selected_encoder_rope_idx, cls_idx=cls_rope_idx, use_cls=True), attn_mask=attn_mask_encoder)
            return z_e
        z_e_AD_t1 = get_encoded_crossview(z_t1_AD, is_selected_AD_cat, is_mask_encoder_AD_cat, num_selected_A, z_AD_encoder_rope_idx, z_AD_cls_rope_idx)  # mini todo lame boundary

        is_mask_D_D = repeat(is_mask_D, 'B N -> (R B) N', R=R)
        z_d_AD_t1_nomaskpos = z_e_AD_t1.new_zeros((R*B_IN,N,D)).masked_scatter(
            repeat(is_selected_D, 'B N -> (R B) N D', R=R, D=D),
            z_e_AD_t1[:, self.in_context_len+num_selected_A:].flatten(),
        )
        z_d = torch.cat([
            z_e_AD_t1[:, :self.in_context_len],
            torch.where(
                repeat(is_mask_D_D, 'RB N -> RB N 1'),
                repeat(self.mask_token.to(dtype), '1 1 D -> (R B) N D', R=R, B=B_IN, N=N),
                z_d_AD_t1_nomaskpos
            )
        ], dim=1)
        z_d = z_d + torch.cat([z_AD_factor_pos.new_zeros((R)*B_IN, self.in_context_len, D), z_AD_factor_pos], dim=1)
        for block_idx, block in enumerate(self.decoder):
            z_d = block(z_d, self.feat_rope_incontext, attn_mask=None)

        # Section: Trains such that decoder used to merging multiple encoder inputs.
        # Tradeoff: This wastes compute by calculating D tokens during A mask. But necessary for constant masks
        # Masking is coupled here, but correct because ultimately is_mask_D valid check and is_mask_A masking combine to work
        z_m = repeat(z_d[:,self.in_context_len:][is_mask_D_D], 'Y D -> (O Y) D', O=O)
        feats_p_rsb = repeat(x_p_D, 'B N D -> (R S B) N D', R=R, S=S)
        feats_p_m = repeat(feats_p_rsb[is_mask_D_D], 'Y D -> (O Y) D', O=O, D=self.patch_dim)
        t_m = torch.rand(z_m.size(0), 1, device=device)
        eps_m = torch.randn_like(feats_p_m)
        feats_p_m_t = t_m * feats_p_m + (1-t_m) * eps_m
        feats_pred = self.diff_mlp(feats_p_m_t, t_m.flatten(), z_m)
        mse_is_valid = repeat(repeat(is_mask_D, 'B N -> (R S B) N', R=R, S=S)[is_mask_D_D], 'Y -> (O Y) 1', O=O) # All true
        mse_loss = 2*(F.huber_loss(feats_pred*mse_is_valid, feats_p_m*mse_is_valid, reduction='none', delta=2) / (1-t_m).clamp_min(0.33).pow(2)).mean() / mse_is_valid.float().mean()

        loss = mse_loss
        return loss


    def forward_reprv2_4_repr_jointenc_crossview(self, x):
        # Merging the first iteration with different targets
        # Goal: include DP repeat
        assert self.in_context_len > 0
        assert self.source_time_repeat == 1
        assert self.repr_noise_bins == 1

        B_IN, C, H, W = x.shape
        device = x.device

        N = self.num_patches
        D = self.hidden_size

        P=self.d_num_p
        R=self.batch_repeat
        R1=self.batch_repeat+1
        O=self.diff_repeat
        S=self.source_time_repeat
        U=self.repr_noise_bins
        V=self.diff_bin_repeat_repr

        def augment_image(x):
            if not self.do_simclr_augs:
                return x, x.new_tensor((0, 0, H, W), dtype=torch.long).repeat(x.size(0), 1)
            x = (x + 1) / 2
            x, random_resize_state = random_resized_crop_batch(x, (H, W))
            x_aug = post_crop_augment_batch(x) * 2 - 1
            return x_aug, random_resize_state

        x_aug_A, x_A_resize_state = augment_image(repeat(x, 'B C H W -> (R B) C H W', R=R))
        x_aug_D, x_D_resize_state = augment_image(x)

        x_p_A = self.patchify(x_aug_A)
        x_p_D = self.patchify(x_aug_D)
        z_A_nopos = self.x_embedder(x_p_A)
        z_D_nopos = self.x_embedder(x_p_D)
        dtype = z_D_nopos.dtype

        is_selected_A, is_mask_A, is_selected_D, is_mask_D, is_selected_Dp, is_mask_Dp, is_selected_AD, is_mask_AD, is_patch_target, is_valid_target = self.get_mask(self.mask_strategy_repr, B_IN, N, device, R=R, check_A_overlap=False)

        is_overlap_A_to_D = visible_a_to_d_patch_overlap(
            is_mask_A, x_A_resize_state, x_D_resize_state,
            H, W, self.patch_size,
        )

        is_selected_DADp = torch.cat([is_selected_D, is_selected_A, is_selected_Dp], 0)
        is_mask_DADp = torch.cat([is_mask_D, is_mask_A, is_mask_Dp], 0)
        is_mask_encoder_DADp = rearrange(is_mask_DADp[is_selected_DADp], '(X m) -> X m', X=(1+R+P)*B_IN)
        is_mask_encoder_AD = rearrange(is_mask_AD[is_selected_AD], '(X m) -> X m', X=R*B_IN)

        def get_factor_pos(is_mask):
            device = is_mask.device
            B_ = is_mask.size(0)
            is_mask_rc = rearrange(is_mask, 'b (h w) -> b h w', h=self.hw_len)
            pos_h = repeat(torch.arange(self.hw_len, device=device), 'h -> b h w', b=B_, w=self.hw_len)
            pos_w = repeat(torch.arange(self.hw_len, device=device), 'w -> b h w', b=B_, h=self.hw_len)
            mean_pos_h = 7 + 0 * ((pos_h*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            mean_pos_w = 7 + 0 * ((pos_w*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            offset_pos_h = rearrange(pos_h, 'b h w -> b (h w)') - mean_pos_h.unsqueeze(-1)
            offset_pos_w = rearrange(pos_w, 'b h w -> b (h w)') - mean_pos_w.unsqueeze(-1)
            offset_idx = (offset_pos_h + self.hw_len - 1 + self.factor_pad) * (2 * self.hw_len - 1 + 2 * self.factor_pad) + (offset_pos_w + self.hw_len - 1 + self.factor_pad)
            factor_pos = repeat(self.factor_posembed.to(dtype), 'f d -> b f d', b=B_).gather(1, repeat(offset_idx, 'b n -> b n d', d=self.hidden_size))
            
            pad_pos_h = repeat(torch.arange(self.pad_hw, device=device), 'h -> b h w', b=B_, w=self.pad_hw)
            pad_pos_w = repeat(torch.arange(self.pad_hw, device=device), 'w -> b h w', b=B_, h=self.pad_hw)
            pad_offset_idx = (pad_pos_h - mean_pos_h[:,None,None] + self.hw_len - 1) * (self.pad_hw + self.hw_len - 1) + (pad_pos_w - mean_pos_w[:,None,None] + self.hw_len - 1)
            factor_pos_pad = repeat(self.factor_posembed.to(dtype), 'f d -> b f d', b=B_).gather(1, repeat(pad_offset_idx, 'b h w -> b (h w) d', d=self.hidden_size))

            patch_rope_idx = rearrange((pos_h + self.factor_pad) * self.pad_hw + (pos_w + self.factor_pad), 'b h w -> b (h w)')
            pad_rope_idx = repeat(torch.arange(self.pad_hw * self.pad_hw, device=device), 'q -> b q', b=B_)
            cls_rope_idx = (mean_pos_h + self.factor_pad) * self.pad_hw + mean_pos_w + self.factor_pad

            return factor_pos, factor_pos_pad, patch_rope_idx, pad_rope_idx, cls_rope_idx
        
        z_DADp_factor_pos, z_DADp_factor_pos_pad, z_DADp_encoder_rope_idx, z_DADp_factorpad_rope_idx, z_DADp_cls_rope_idx = get_factor_pos(is_mask_DADp)
        z_AD_factor_pos, z_AD_factor_pos_pad, z_AD_encoder_rope_idx, z_AD_factorpad_rope_idx, z_AD_cls_rope_idx = get_factor_pos(is_mask_AD)

        z_t1_DADp = torch.cat([
            z_D_nopos,
            z_A_nopos,
            torch.where(
                rearrange(is_patch_target, 'P B N -> (P B) N 1'), 
                repeat(z_D_nopos, "B N D -> (P B) N D", P=P) + rearrange(self.is_d_embed.to(dtype), '1 D -> 1 1 D'),
                repeat(z_D_nopos, 'B N D -> (P B) N D', P=P),
            )
        ], dim=0) + z_DADp_factor_pos   
        def get_encoded(z, is_selected, is_mask_encoder):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, D)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic, "separate")
            cls_idx = torch.full((B_,), ((self.hw_len-1)//2)*(self.hw_len+1), device=device, dtype=torch.long)  # Todo: wrong for factorpad
            for block_idx, block in enumerate(self.encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, cls_idx=cls_idx, use_cls=True), attn_mask=attn_mask_encoder)
            return z_e
        z_e_DADp_t1 = get_encoded(z_t1_DADp, is_selected_DADp, is_mask_encoder_DADp)

        z_t1_AD = torch.cat([
            z_A_nopos + z_AD_factor_pos,
            repeat(z_D_nopos, 'B N D -> (R B) N D', R=R) + z_AD_factor_pos,
        ], dim=1)
        is_selected_AD_cat = torch.cat([is_selected_A, repeat(is_selected_D, 'B N -> (R B) N', R=R)], dim=1)
        is_mask_AD_cat = torch.cat([is_mask_A, repeat(is_mask_D, 'B N -> (R B) N', R=R)], dim=1)
        is_mask_encoder_AD_cat = rearrange(is_mask_AD_cat[is_selected_AD_cat], '(X m) -> X m', X=R*B_IN)
        num_selected_A = is_selected_A[0].sum()
        assert (is_selected_A.sum(1) == num_selected_A).all()
        def get_encoded_crossview(z, is_selected, is_mask_encoder, view_boundary, encoder_rope_idx, cls_rope_idx):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, D)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic, "separate")
            attn_mask_encoder_viewseparate = attn_mask_encoder.clone()
            attn_mask_encoder_viewseparate[:,:,self.in_context_len+view_boundary:,self.in_context_len:self.in_context_len+view_boundary] = False
            attn_mask_encoder_viewseparate[:,:,self.in_context_len:self.in_context_len+view_boundary,self.in_context_len+view_boundary:] = False
            # first todo: rope indices
            # second todo: alternate masking (vggt)
            # giga todo: Factor pad broken, but that opens a whole can of worms
            selected_encoder_rope_idx = torch.cat([encoder_rope_idx, encoder_rope_idx], dim=1)[is_selected].view(B_, -1)
            for block_idx, block in enumerate(self.encoder):
                if block_idx % 2 == 0:
                    z_e = block(z_e, partial(self.feat_rope_incontext_pad, idx=selected_encoder_rope_idx, cls_idx=cls_rope_idx, use_cls=True), attn_mask=attn_mask_encoder_viewseparate)
                else:
                    z_e = block(z_e, partial(self.feat_rope_incontext_pad, idx=selected_encoder_rope_idx, cls_idx=cls_rope_idx, use_cls=True), attn_mask=attn_mask_encoder)
            return z_e
        z_e_AD_t1 = get_encoded_crossview(z_t1_AD, is_selected_AD_cat, is_mask_encoder_AD_cat, num_selected_A, z_AD_encoder_rope_idx, z_AD_cls_rope_idx)  # mini todo lame boundary

        is_mask_D_D = repeat(is_mask_D, 'B N -> (R1 B) N', R1=R1)
        z_d_D_t1_nomaskpos = z_e_DADp_t1.new_zeros((B_IN,N,D)).masked_scatter(
            repeat(is_selected_D, 'B N -> B N D', D=D),
            z_e_DADp_t1[:B_IN, self.in_context_len:].flatten(),
        )
        z_d_AD_t1_nomaskpos = z_e_AD_t1.new_zeros((R*B_IN,N,D)).masked_scatter(
            repeat(is_selected_D, 'B N -> (R B) N D', R=R, D=D),
            z_e_AD_t1[:, self.in_context_len+num_selected_A:].flatten(),
        )
        z_d_DAD_t1_nomaskpos = torch.cat([z_d_D_t1_nomaskpos, z_d_AD_t1_nomaskpos], dim=0)
        z_d = torch.cat([
            torch.cat([z_e_DADp_t1[:B_IN, :self.in_context_len], z_e_AD_t1[:, :self.in_context_len]], dim=0),
            torch.where(
                repeat(is_mask_D_D, 'RB N -> RB N 1'),
                repeat(self.mask_token.to(dtype), '1 1 D -> (R B) N D', R=1+R, B=B_IN, N=N),
                z_d_DAD_t1_nomaskpos
            )
        ], dim=1)
        z_DAD_factor_pos = torch.cat([z_DADp_factor_pos[:B_IN], z_AD_factor_pos], dim=0)
        z_d = z_d + torch.cat([z_DAD_factor_pos.new_zeros((1+R)*B_IN, self.in_context_len, D), z_DAD_factor_pos], dim=1)
        for block_idx, block in enumerate(self.decoder):
            z_d = block(z_d, self.feat_rope_incontext, attn_mask=None)

        # Section: Trains such that decoder used to merging multiple encoder inputs.
        # Tradeoff: This wastes compute by calculating D tokens during A mask. But necessary for constant masks
        # Masking is coupled here, but correct because ultimately is_mask_D valid check and is_mask_A masking combine to work
        z_m = repeat(z_d[:,self.in_context_len:][is_mask_D_D], 'Y D -> (O Y) D', O=O)
        feats_p_rsb = repeat(x_p_D, 'B N D -> (R S B) N D', R=R1, S=S)
        feats_p_m = repeat(feats_p_rsb[is_mask_D_D], 'Y D -> (O Y) D', O=O, D=self.patch_dim)
        t_m = torch.rand(z_m.size(0), 1, device=device)
        eps_m = torch.randn_like(feats_p_m)
        feats_p_m_t = t_m * feats_p_m + (1-t_m) * eps_m
        feats_pred = self.diff_mlp(feats_p_m_t, t_m.flatten(), z_m)
        mse_is_valid = repeat(repeat(is_mask_D, 'B N -> (R S B) N', R=R1, S=S)[is_mask_D_D], 'Y -> (O Y) 1', O=O) # MINI TODO All true
        mse_loss = 2*(F.huber_loss(feats_pred*mse_is_valid, feats_p_m*mse_is_valid, reduction='none', delta=2) / (1-t_m).clamp_min(0.33).pow(2)).mean() / mse_is_valid.float().mean()
        
        with torch.no_grad():
            z_d_prs = repeat(z_d[:,self.in_context_len:], '(R S B) N D -> P R S B N D', P=P, R=R1, S=S).float()
            is_dp_prs = repeat(is_patch_target, 'P B N -> P R S B N', R=R1, S=S)
            z_dp_uvprs = repeat(z_d_prs[is_dp_prs], '(X) D -> (U V X) D', U=U, V=V) # (U V P R S B) D

            x_dp_uvprs_t1 = repeat(repeat(x_p_D, 'B N D -> P B N D', P=P)[is_patch_target], '(P B) D -> (U V P R S B) D', P=P, R=R1, S=S, U=U, V=V).float()
            tu_dp_uvprs = torch.rand(U,V,P,S,B_IN, device=device) * (self.noise_bin_limits[1:]-self.noise_bin_limits[:-1]).float().view(-1,1,1,1,1) + self.noise_bin_limits[:-1].view(-1,1,1,1,1)
            tu_dp_uvprs = repeat(tu_dp_uvprs, 'U V P S B -> (U V P R S B) 1', R=R1).float()
            eps_dp_uvprs_tu = repeat(torch.randn(U,V,P,S,B_IN,self.patch_dim,device=device), 'U V P S B D -> (U V P R S B) D', R=R1).float()
            x_dp_uvprs_tu = tu_dp_uvprs * x_dp_uvprs_t1 + (1-tu_dp_uvprs) * eps_dp_uvprs_tu

            x_dp_qprs_pred = self.diff_mlp(x_dp_uvprs_tu.to(dtype), tu_dp_uvprs.flatten().to(dtype), z_dp_uvprs.to(dtype)).float()

            freq_weight = 1/2*tu_dp_uvprs/(1-tu_dp_uvprs).clamp_min(0.33).pow(2)
            h_p_uvprb_pred_only = freq_weight * (x_dp_qprs_pred - x_dp_uvprs_t1).pow(2).sum(-1, True)
            h_p_rbpuv_pred_only = rearrange(h_p_uvprb_pred_only, '(U V P R B) 1 -> R B P U V', P=P, R=R1, U=U, V=V).to(dtype)

        # REPR Calculation
        def get_repr(z_e, repr_factor_pos_pad, factorpad_rope_idx, encoder_rope_idx, cls_rope_idx, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)
            
            same_rope = rearrange(factorpad_rope_idx, 'B Q -> B Q 1') == rearrange(encoder_rope_idx, 'B K -> B 1 K')
            same_rope_or_in_context = F.pad(same_rope, (self.in_context_len, 0), value=False)
            q_has_k = reduce(same_rope, 'B Q K -> B Q 1', 'any')
            q_is_restricted = q_has_k & (torch.rand_like(q_has_k, dtype=dtype) < 0.0)
            attn_mask_default = rearrange(~is_mask_encoder_ic, 'B K -> B 1 K')
            attn_mask_restricted = rearrange(~is_mask_encoder_ic, 'B K -> B 1 K') & ((~q_is_restricted) | same_rope_or_in_context)
            attn_mask = repeat(torch.where(q_is_restricted, attn_mask_restricted, attn_mask_default), 'B Q K -> B H Q K', H=self.num_heads)
            
            repr_tokens = repr_factor_pos_pad
            q_rope = partial(self.feat_rope_incontext_pad, idx=factorpad_rope_idx, cls_idx=None, use_cls=False)
            k_rope = partial(self.feat_rope_incontext_pad, idx=encoder_rope_idx, cls_idx=cls_rope_idx, use_cls=True)
            for block in self.representation_encoder:
                repr_tokens = block(repr_tokens, z_e, q_rope, k_rope, attn_mask=attn_mask)
            return repr_tokens

        z_DADp_t1_repr = get_repr(z_e_DADp_t1, z_DADp_factor_pos_pad, z_DADp_factorpad_rope_idx, z_DADp_encoder_rope_idx[is_selected_DADp].view((1+R+P)*B_IN,-1), z_DADp_cls_rope_idx, is_mask_encoder_DADp)
        z_DA_t1_repr = repeat(self.repr_final_layer(z_DADp_t1_repr[:(1+R)*B_IN]), '(R b) q d -> R b P q d', R=R1, P=P)
        z_Dp_t1_repr = repeat(self.repr_final_layer(z_DADp_t1_repr[(1+R)*B_IN:]), '(P b) q (U d) -> b P U q d', P=P, U=U)
        z_t1_repr_global = repeat(self.repr_final_layer_global(z_e_DADp_t1[:(1+R)*B_IN, :self.in_context_len]), '(R b) C d -> R b P U (C d)', R=R1, P=P, U=U)
        z_Dp_t1_repr_global = repeat(self.repr_final_layer_global(z_e_DADp_t1[(1+R)*B_IN:, :self.in_context_len]), '(P b) C d -> b P U (C d)', P=P, U=U)

        z_DA_t1_repr = torch.where(torch.rand(z_DA_t1_repr.size(0),device=device).view(-1,1,1,1,1) < self.pad_dropout, z_DA_t1_repr*self.pad_dropout_mask.to(dtype), z_DA_t1_repr)

        is_mask_DA = torch.cat([is_mask_D, is_mask_A], dim=0)
        z_DA_t1_repr = torch.where(reduce(is_mask_DA, '(R b) n -> R b 1 1 1', 'all', R=R1), 0, z_DA_t1_repr) # Special case: zero out (prevents magnitude drift)
        z_t1_repr_global = torch.where(reduce(is_mask_DA, '(R b) n -> R b 1 1 1', 'all', R=R1), 0, z_t1_repr_global)
        z_DA_t1_repr = repeat(z_DA_t1_repr, 'R b P q D -> R b P U q D', U=U) # R b P U q d
        z_D_t1_repr = z_DA_t1_repr[:1]
        z_A_t1_repr = z_DA_t1_repr[1:]
        z_D_t1_repr_global = z_t1_repr_global[:1]
        z_A_t1_repr_global = z_t1_repr_global[1:]

        is_valid_pmi = reduce(~(repeat(~is_valid_target, '(R b) n -> R b P n', R=R1, P=P) & repeat(is_patch_target, 'P b n -> R b P n', R=R1)), 'R b P n -> R b P 1', 'all')  # don't allow loss for: ~is_mask and is_dp. gross rn, is jsut is valid d
        is_valid_D = is_valid_pmi[:1]
        is_valid_A = reduce(~(repeat(is_overlap_A_to_D, '(R b) n -> R b P n', R=R, P=P) & repeat(is_patch_target, 'P b n -> R b P n', R=R)), 'R b P n -> R b P 1', 'all')
        is_valid = is_valid_D & is_valid_A

        h_p_given_D_preintegration = h_p_rbpuv_pred_only[:1]
        h_p_given_DA_preintegration = h_p_rbpuv_pred_only[1:]

        pmi_pred = reduce(z_A_t1_repr * (z_Dp_t1_repr - z_D_t1_repr) / 2 * z_A_t1_repr.size(-2)**0.5, 'R b P U q d -> R b P U', 'mean') * D**0.5  # GIGA TODO USE REPR HERE!
        pmi_pred_global = reduce(z_A_t1_repr_global * (z_Dp_t1_repr_global - z_D_t1_repr_global) / 2, 'R b P U d -> R b P U', 'mean') * z_A_t1_repr_global.size(-1)**0.5
        pmi_target = (h_p_given_D_preintegration - h_p_given_DA_preintegration).clamp(-75,75).mean(-1) * self.pmi_target_mul  # GIGA TODO THIS CLAMPING IS ONLY GOOD FOR PIXELS (BUT SANCHECKED)

        # Factorization has an optimum at zero, so huber should be fine, no?
        repr_loss = (2*F.huber_loss(pmi_pred, pmi_target, reduction='none', delta=2) * is_valid / reduce(is_valid.bfloat16(), 'R b P U -> R b 1 1', 'mean').clamp_min(1e-5)).mean()
        repr_loss_global = (2*F.huber_loss(pmi_pred_global, pmi_target, reduction='none', delta=2) * is_valid / reduce(is_valid.bfloat16(), 'R b P U -> R b 1 1', 'mean').clamp_min(1e-5)).mean()

        loss = mse_loss + self.repr_loss_weight * repr_loss + self.repr_loss_weight * repr_loss_global
        return loss, mse_loss.detach().clone(), repr_loss.detach().clone(), repr_loss_global.detach().clone()


    @torch.compile
    def forward_reprv2_4_feats(self, x, num_mask_tokens=0, is_mask_rb=None, return_padded=True, labels=None):
        assert self.in_context_len > 0
        if self.y_embedder is not None and labels is None:
            raise ValueError("Class labels are required when condition_on_class=True")

        B_IN, C, H, W = x.shape
        device = x.device
        x_p_og = self.patchify(x)
        z_t1_og_nopos = self.x_embedder(x_p_og)
        dtype = z_t1_og_nopos.dtype

        N = self.num_patches
        D = self.hidden_size
        R=1

        if is_mask_rb is None:
            mask_rank = torch.rand((B_IN, N), device=device).argsort(dim=1).argsort(dim=1)
            is_mask_rb = mask_rank < num_mask_tokens
            is_selected_rb = ~is_mask_rb
        else:
            is_mask_rb = is_mask_rb.to(device=device, dtype=torch.bool)
            max_selected = (~is_mask_rb).sum(dim=1).max()
            is_selected_rb = is_mask_rb.int().argsort(dim=1).argsort(dim=1) < max_selected
        is_mask_encoder_rb = rearrange(is_mask_rb[is_selected_rb], '(B m) -> B m', B=B_IN)

        def get_factor_pos(is_mask):
            device = is_mask.device
            B_ = is_mask.size(0)
            is_mask_rc = rearrange(is_mask, 'b (h w) -> b h w', h=self.hw_len)
            pos_h = repeat(torch.arange(self.hw_len, device=device), 'h -> b h w', b=B_, w=self.hw_len)
            pos_w = repeat(torch.arange(self.hw_len, device=device), 'w -> b h w', b=B_, h=self.hw_len)
            mean_pos_h = ((pos_h*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            mean_pos_w = ((pos_w*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            offset_pos_h = rearrange(pos_h, 'b h w -> b (h w)') - mean_pos_h.unsqueeze(-1)
            offset_pos_w = rearrange(pos_w, 'b h w -> b (h w)') - mean_pos_w.unsqueeze(-1)
            offset_idx = (offset_pos_h + self.hw_len - 1 + self.factor_pad) * (2 * self.hw_len - 1 + 2 * self.factor_pad) + (offset_pos_w + self.hw_len - 1 + self.factor_pad)
            factor_pos = repeat(self.factor_posembed.to(dtype), 'f d -> b f d', b=B_).gather(1, repeat(offset_idx, 'b n -> b n d', d=self.hidden_size))
            
            pad_pos_h = repeat(torch.arange(self.pad_hw, device=device), 'h -> b h w', b=B_, w=self.pad_hw)
            pad_pos_w = repeat(torch.arange(self.pad_hw, device=device), 'w -> b h w', b=B_, h=self.pad_hw)
            pad_offset_idx = (pad_pos_h - mean_pos_h[:,None,None] + self.hw_len - 1) * (self.pad_hw + self.hw_len - 1) + (pad_pos_w - mean_pos_w[:,None,None] + self.hw_len - 1)
            factor_pos_pad = repeat(self.factor_posembed.to(dtype), 'f d -> b f d', b=B_).gather(1, repeat(pad_offset_idx, 'b h w -> b (h w) d', d=self.hidden_size))

            patch_rope_idx = rearrange((pos_h + self.factor_pad) * self.pad_hw + (pos_w + self.factor_pad), 'b h w -> b (h w)')
            pad_rope_idx = repeat(torch.arange(self.pad_hw * self.pad_hw, device=device), 'q -> b q', b=B_)
            cls_rope_idx = (mean_pos_h + self.factor_pad) * self.pad_hw + mean_pos_w + self.factor_pad

            if return_padded:
                return factor_pos, factor_pos_pad, patch_rope_idx, pad_rope_idx, cls_rope_idx
            else:
                return factor_pos, factor_pos, patch_rope_idx, patch_rope_idx, cls_rope_idx
        
        z_factor_pos, z_factor_pos_pad, z_patch_rope_idx, z_pad_rope_idx, z_cls_rope_idx = get_factor_pos(is_mask_rb)

        z_t1 = repeat(z_t1_og_nopos, 'B N D -> (R B) N D', R=R) + z_factor_pos
                
        def get_encoded(z, is_selected, is_mask_encoder):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, D)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            if self.y_embedder is not None:
                z_e[:, :self.in_context_len] += self.y_embedder(labels)[:, None].to(dtype)
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic, "separate")
            for block_idx, block in enumerate(self.encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
        
        z_e_t1 = get_encoded(z_t1, is_selected_rb, is_mask_encoder_rb)

        def get_repr_encoder(z_e, is_selected, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic, "separate")
            for block_idx, block in enumerate(self.repr_encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e

        if self.is_repr_encoderdecoder:
            z_re_t1 = get_repr_encoder(z_e_t1, is_selected_rb, is_mask_encoder_rb)
        else:
            z_re_t1 = z_e_t1

        def get_repr(z_e, repr_factor_pos_pad, factorpad_rope_idx, encoder_rope_idx, cls_rope_idx, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)
            
            same_rope = rearrange(factorpad_rope_idx, 'B Q -> B Q 1') == rearrange(encoder_rope_idx, 'B K -> B 1 K')
            same_rope_or_in_context = F.pad(same_rope, (self.in_context_len, 0), value=False)
            q_has_k = reduce(same_rope, 'B Q K -> B Q 1', 'any')
            q_is_restricted = q_has_k & (torch.rand_like(q_has_k, dtype=dtype) < 0.0)
            attn_mask_default = rearrange(~is_mask_encoder_ic, 'B K -> B 1 K')
            attn_mask_restricted = rearrange(~is_mask_encoder_ic, 'B K -> B 1 K') & ((~q_is_restricted) | same_rope_or_in_context)
            attn_mask = repeat(torch.where(q_is_restricted, attn_mask_restricted, attn_mask_default), 'B Q K -> B H Q K', H=self.num_heads)
            
            repr_tokens = repr_factor_pos_pad
            q_rope = partial(self.feat_rope_incontext_pad, idx=factorpad_rope_idx, cls_idx=None, use_cls=False)
            k_rope = partial(self.feat_rope_incontext_pad, idx=encoder_rope_idx, cls_idx=cls_rope_idx, use_cls=True)
            for block in self.representation_encoder:
                repr_tokens = block(repr_tokens, z_e, q_rope, k_rope, attn_mask=attn_mask)
            return repr_tokens

        z_t1_repr = self.repr_final_layer(get_repr(z_re_t1, z_factor_pos_pad, z_pad_rope_idx, z_patch_rope_idx[is_selected_rb].view(R*B_IN,-1), z_cls_rope_idx, is_mask_encoder_rb))
        z_t1_repr_global = rearrange(self.repr_final_layer_global(z_re_t1[:, :self.in_context_len]), 'RB C d -> RB (C d)')

        return z_t1_repr, z_t1_repr_global

    @torch.compile
    def forward_noised_mar_frozen_feats_imagenet_labels(self, x, feats, labels):
        # SLOP CODE
        assert self.in_context_len > 0

        B_IN, C, H, W = x.shape
        device = x.device
        x_p_og = self.patchify(x)
        feats = feats.detach().to(device=device, dtype=x_p_og.dtype)

        B = B_IN * self.batch_repeat
        N = self.num_patches
        D = self.hidden_size

        is_mask_stage_1, is_mask_stage_2 = self.sample_m_mar(B,N,device)
        is_mask_encoder = rearrange(is_mask_stage_2[~is_mask_stage_1], '(b m) -> b m', b=B)
        M = is_mask_encoder.size(1)
        t_e = torch.rand(B, 1, 1, device=device) * torch.rand(B, 1, 1, device=device)

        x_p = x_p_og.repeat(self.batch_repeat, 1, 1)
        feats = feats.repeat(self.batch_repeat, 1, 1) if feats.size(0) == B_IN else feats
        x_p_eps = torch.randn_like(x_p)
        feats_eps = torch.randn_like(feats)
        input_noise_mask_pixels = (torch.rand(B, N, 1, device=device) < 0.05)
        x_p_t = torch.where(input_noise_mask_pixels, t_e * x_p + (1-t_e) * x_p_eps, x_p)
        input_noise_mask_feats = (torch.rand(B, N, 1, device=device) < 0.05)
        feats_t = torch.where(input_noise_mask_feats, t_e * feats + (1-t_e) * feats_eps, torch.zeros_like(feats))
        z = self.x_embedder(x_p_t) + self.feats_embedder(feats_t)
        dtype = z.dtype

        z_e = z[~is_mask_stage_1].view(B, M, D)  # e: encoder
        z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B), z_e], dim=1)
        is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B, self.in_context_len), is_mask_encoder], dim=1)
        attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic, "separate")
        c_e = self.t_embedder(t_e.flatten()) # B D
        for block_idx, block in enumerate(self.encoder):
            z_e = block(z_e, c_e, partial(self.feat_rope_incontext, m=is_mask_stage_1), attn_mask=attn_mask_encoder)

        imagenet_labels = labels.long().repeat(self.batch_repeat)
        imagenet_logits = self.imagenet_probe(z_e[:, 0]).float()
        imagenet_loss = F.cross_entropy(imagenet_logits, imagenet_labels)
        imagenet_acc = (imagenet_logits.argmax(dim=-1) == imagenet_labels).float().mean()

        z_d = torch.cat([
            z_e[:, :self.in_context_len],
            self.mask_token.to(dtype).repeat(B,N,1).masked_scatter(
                (~is_mask_stage_2).unsqueeze(-1).expand_as(z),
                z_e[:, self.in_context_len:][~is_mask_encoder].flatten()
            )
        ], dim=1)
        for block_idx, block in enumerate(self.decoder):
            z_d = block(z_d, self.feat_rope_incontext)

        z_m = repeat(z_d[:,self.in_context_len:][is_mask_stage_2], 'Q D -> (R Q) D', R=self.diff_repeat)
        x_p_m = repeat(x_p[is_mask_stage_2], 'Q D -> (R Q) D', R=self.diff_repeat, D=self.patch_dim)
        t_m = torch.rand(z_m.size(0), 1, device=device)
        eps_m = torch.randn_like(x_p_m)
        x_p_m_t = t_m * x_p_m + (1-t_m) * eps_m
        x_pred = self.diff_mlp(x_p_m_t, t_m.flatten(), z_m)
        mar_loss = 2*F.huber_loss(x_pred, x_p_m)

        feats_m = repeat(feats[is_mask_stage_2], 'Q D -> (R Q) D', R=self.diff_repeat, D=self.repr_dim)
        t_feats_m = torch.rand(z_m.size(0), 1, device=device)
        eps_feats_m = torch.randn_like(feats_m)
        feats_m_t = t_feats_m * feats_m + (1-t_feats_m) * eps_feats_m
        feats_pred = self.diff_mlp_feats(feats_m_t, t_feats_m.flatten(), z_m)
        mar_loss_feats = F.mse_loss(feats_pred, feats_m)

        loss = mar_loss + 10 * mar_loss_feats + 0.1*imagenet_loss

        return loss, mar_loss.detach().clone(), mar_loss_feats.detach().clone(), imagenet_loss.detach().clone(), imagenet_acc.detach().clone()

    def forward_repr_20(self, x):
        loss, mse_loss, repr_loss, repr_loss_global = self.forward_reprv2_3_repr_multinoise(x)
        return loss, {
            'mse_loss': mse_loss.item(),
            'repr_loss': repr_loss.item(),
            'repr_loss_global': repr_loss_global.item(),
        }

    def forward_repr_22(self, x, x_target=None, labels=None):
        loss, mse_loss, repr_loss, repr_loss_global = self.forward_reprv2_4_repr_jointenc(x, x_target, labels=labels)
        return loss, {
            'mse_loss': mse_loss.item(),
            'repr_loss': repr_loss.item(),
            'repr_loss_global': repr_loss_global.item(),
        }

    def forward_repr_mar_in_ft(self, x, labels):
        loss, mar_loss, imagenet_loss, imagenet_acc = self.forward_noised_mar_imagenet_labels(x, labels)
        return loss, {
            'mar_loss': mar_loss.item(),
            'imagenet_loss': imagenet_loss.item(),
            'imagenet_acc': imagenet_acc.item(),
        }

    def forward_repr_mar_frozen_feats_imagenet_labels(self, x, feats, labels):
        loss, mar_loss, mar_loss_feats, imagenet_loss, imagenet_acc = self.forward_noised_mar_frozen_feats_imagenet_labels(x, feats, labels)
        return loss, {
            'mar_loss': mar_loss.item(),
            'mar_loss_feats': mar_loss_feats.item(),
            'imagenet_loss': imagenet_loss.item(),
            'imagenet_acc': imagenet_acc.item(),
        }

def REPR_XT_16(**kwargs):
    return REPR(depth=18, hidden_size=256, num_heads=4,
               patch_size=16, decoder_depth=6, **kwargs)

def REPR_T_16(**kwargs):
    return REPR(depth=18, hidden_size=384, num_heads=6,
               patch_size=16, decoder_depth=6, **kwargs)

def REPR_S_16(**kwargs):
    return REPR(depth=18, hidden_size=512, num_heads=8,
               patch_size=16, decoder_depth=6, **kwargs)

def REPR_B_16(**kwargs):
    return REPR(depth=18, hidden_size=768, num_heads=12,
               patch_size=16, decoder_depth=6, **kwargs)

def REPR_L_16(**kwargs):
    return REPR(depth=18, hidden_size=1024, num_heads=16,
               patch_size=16, decoder_depth=6, **kwargs)

REPR_models = {
    'REPR-XT/16': REPR_XT_16,
    'REPR-T/16': REPR_T_16,
    'REPR-S/16': REPR_S_16,
    'REPR-B/16': REPR_B_16,
    'REPR-L/16': REPR_L_16,
}
