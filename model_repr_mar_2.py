# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from einops import rearrange, repeat, reduce


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
        theta = 10000,
        num_cls_token = 0
    ):
        super().__init__()
        freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = torch.arange(pt_seq_len) / pt_seq_len * pt_seq_len

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
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
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
    """Joint masked image modeling and representation learning."""
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        in_context_len=16,
        repr_dim=768,
        decoder_depth=6,
        diff_mlp_depth=3,
        batch_repeat=1,
        batch_repeat_mar=1,
        diff_repeat=4,
        b_num_p=8,
        diff_repeat_repr=32,
        b_max=48,
        mask_m=256 - 76,
        mask_strategy_mar='block',
        mask_strategy_repr='block',
        repr_loss_weight=0.1,
        repr_delta=0.2,
        pad_dropout=0.05,
        mu_repr=-1.2,
        std_repr=1.2,
        mu_mar=-1.2,
        std_mar=1.2,
        is_first_iter=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.in_context_len = in_context_len
        self.repr_dim = repr_dim
        self.batch_repeat = batch_repeat
        self.batch_repeat_mar = batch_repeat_mar
        self.diff_repeat = diff_repeat
        self.mask_m = mask_m
        self.mask_strategy_mar = mask_strategy_mar
        self.mask_strategy_repr = mask_strategy_repr
        self.repr_loss_weight = repr_loss_weight
        self.repr_delta = repr_delta
        self.pad_dropout = pad_dropout
        self.is_first_iter = is_first_iter
        self.mu_repr, self.std_repr, self.mu_mar, self.std_mar = mu_repr, std_repr, mu_mar, std_mar

        # linear embed
        self.x_embedder = nn.Linear(self.patch_dim, hidden_size, bias=True)

        if self.in_context_len > 0:
            self.in_context_tokens = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_tokens, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # transformer
        self.encoder = nn.ModuleList([
            MAGEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for i in range(depth)
        ])

        self.decoder = nn.ModuleList([
            MAGEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for i in range(decoder_depth)
        ])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.b_num_p = b_num_p
        self.b_max = b_max
        self.hw_len = input_size // patch_size
        self.diff_repeat_repr = diff_repeat_repr

        self.factor_pad = 1
        self.pad_hw = self.hw_len + 2*self.factor_pad
        pad_dropout_mask = torch.zeros(self.pad_hw, self.pad_hw)
        pad_dropout_mask[self.factor_pad:self.factor_pad+self.hw_len,self.factor_pad:self.factor_pad+self.hw_len] = 1
        self.register_buffer('pad_dropout_mask', pad_dropout_mask.flatten()[...,None])
        self.factor_posembed = nn.Parameter(torch.zeros((hw_seq_len*2-1+2*self.factor_pad)**2, hidden_size), requires_grad=True)
        torch.nn.init.normal_(self.factor_posembed, std=.02)
        self.feat_rope_incontext_pad = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len + 2*self.factor_pad,
            num_cls_token=self.in_context_len
        )
        self.repr_final_layer = FinalLayerLearnedModulation(hidden_size, repr_dim)
        self.repr_final_layer_global = FinalLayerLearnedModulation(hidden_size, repr_dim)
        self.representation_encoder = nn.ModuleList([
            RepresentationCrossAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(3)
        ])


        self.is_b_embed = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        torch.nn.init.normal_(self.is_b_embed, std=.02)

        self.repr_encoder = nn.ModuleList([
            MAGEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for i in range(3)
        ])

        if self.is_first_iter:
            self.diff_mlp = SimpleMLPAdaLN(self.patch_dim, hidden_size, self.patch_dim, hidden_size, diff_mlp_depth)
        else:
            self.feats_embedder = nn.Linear(self.repr_dim, hidden_size, bias=True)
            self.diff_mlp = SimpleMLPAdaLN(self.repr_dim, hidden_size, self.repr_dim, hidden_size, diff_mlp_depth)

        self.initialize_weights()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        old_key = prefix + "is_d_embed"
        if old_key in state_dict:
            state_dict.setdefault(prefix + "is_b_embed", state_dict.pop(old_key))
        state_dict.pop(prefix + "noise_bin_limits", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
    
    def get_attention_mask(self, m):
        k = m.unsqueeze(1)
        q = m.unsqueeze(2)
        return (q == k).unsqueeze(1)


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

    def sample_mask_random(self, B_IN, N, device, H=16, W=16, R=1):
        assert H * W == N

        P = self.b_num_p
        b_max = min(self.b_max, N)

        def random_mask(B, min_masked):
            counts = self._balanced_mask_counts(B, N, min_masked, device)
            rank = torch.rand(B, N, device=device).argsort(dim=1).argsort(dim=1)
            return rank < counts[:, None]

        is_mask_A = random_mask(R * B_IN, self.mask_m)
        is_mask_B = random_mask(B_IN, N - b_max)

        is_valid_A = rearrange(is_mask_A, '(R B) N -> R B N', R=R).all(dim=0)
        is_valid_bp = is_valid_A & is_mask_B
        is_bp_idx = (torch.rand(B_IN, N, device=device) + is_valid_bp.float()).argsort(dim=1, descending=True)[:, :P]
        is_bp = torch.zeros(P * B_IN, N, device=device, dtype=bool).scatter(1, rearrange(is_bp_idx, 'B P -> (P B) 1'), True)
        is_bp = rearrange(is_bp, '(P B) N -> P B N', P=P)

        is_mask_Bp = repeat(is_mask_B, 'B N -> P B N', P=P) & (~is_bp)

        def get_selected(mask, num_selected):
            rank = torch.rand_like(mask.float()).masked_fill(~mask, -1).argsort(dim=1)
            return torch.zeros_like(mask).scatter(1, rank[:, :num_selected], True)

        num_selected_ab = max(N - self.mask_m, self.b_max + 1)
        is_selected_A = get_selected(is_mask_A, num_selected_ab)
        is_selected_B = get_selected(is_mask_B, num_selected_ab)
        is_selected_Bp = get_selected(rearrange(is_mask_Bp, 'P B N -> (P B) N'), num_selected_ab)
        is_mask_AB = is_mask_A & repeat(is_mask_B, 'B N -> (R B) N', R=R)
        is_selected_AB = get_selected(is_mask_AB, N - self.mask_m + self.b_max)
        is_valid_target = torch.cat([is_mask_B, is_mask_A], dim=0)

        return is_selected_A, is_mask_A, is_selected_B, is_mask_B, is_selected_Bp, rearrange(is_mask_Bp, 'P B N -> (P B) N'), is_selected_AB, is_mask_AB, is_bp, is_valid_target

    def sample_mask_block(self, B_IN, N, device, H=16, W=16, R=1):
        # Guarantees 
        # m_index 1 has min_mask_ratio selected
        # m_index 2 is uniform over [min_mask_ratio, 1] per batch index.
        #   Global sum equal for compile
        # is_mask_background_2: background without B, shape b,n
        # is_mask_background_2_bp: background without BP, exactly 1 fewer true per element, shape d,b,n
        # is_mask_background_1_bp: background without BP MAE encoder mask  TODO clever 2x speedup.
        
        def block_mask(B, N, M, block=4):
            # Slop code that works
            assert H * W == N and 0 <= M <= N and block >= 1

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

        P = self.b_num_p

        is_mask_encoder = block_mask(R*B_IN,N,self.mask_m)

        is_valid_mask = rearrange(is_mask_encoder, 'RB (H W) -> RB H W', H=H, W=W).clone()
        is_valid_target = is_valid_mask.clone()
        is_valid_target = rearrange(is_valid_target, 'RB H W -> RB (H W)')

        is_bp_idx = (repeat(~is_valid_target, '(R B) N -> R B N', R=R).sum(dim=0) + torch.rand((B_IN,N), device=device) * 0.5).argsort(dim=1)[:,:P]
        is_bp = torch.zeros(P*B_IN, N, device=device, dtype=bool).scatter(1, rearrange(is_bp_idx, 'b p -> (p b) 1'), 1)
        is_bp = rearrange(is_bp, '(P B) N -> P B N', P=P)

        b_max = min(self.b_max, N)
        k = torch.randint(0, b_max + 1, (B_IN, 1), device=device)  # how many False per row
        is_background_idx = (torch.rand(B_IN, N, device=device) + is_bp.any(0)).argsort(dim=1)[:, :b_max]  # 0% chance take is_bp
        is_mask_background = torch.ones(B_IN, N, dtype=torch.bool, device=device).scatter(1, is_background_idx, torch.arange(b_max, device=device).unsqueeze(0) >= k)
        is_mask_background_bp = repeat(is_mask_background, 'B N -> P B N', P=P) & (~is_bp)
        is_selected_background_bp = torch.zeros(P, B_IN, N, dtype=bool, device=device).scatter(2, is_mask_background_bp.int().argsort(2)[:,:,:b_max+1], True)
        
        is_selected_background = is_selected_background_bp[0]  # Any element works
        is_selected_background_bp = rearrange(is_selected_background_bp, 'P B N -> (P B) N')
        is_mask_background_bp = rearrange(is_mask_background_bp, 'P B N -> (P B) N')

        is_mask = torch.cat([is_mask_background,is_mask_encoder],dim=0)  # GIGA TODO REQUIRES IS B < IS M
        is_selected_encoder = torch.zeros_like(is_mask).scatter(1, torch.rand_like(is_mask.float()).masked_fill(~is_mask, -1).argsort(dim=1)[:, : N - self.mask_m], True)
        is_valid_target = torch.cat([is_mask_background, is_valid_target], dim=0)


        def get_selected(mask, num_selected):
            return torch.zeros_like(mask).scatter(1, torch.rand_like(mask.float()).masked_fill(~mask, -1).argsort(dim=1)[:, : num_selected], True)
        
        num_selected_ab = max(N-self.mask_m, self.b_max+1)
        is_mask_A = is_mask_encoder
        is_selected_A = get_selected(is_mask_encoder, num_selected_ab)
        is_mask_B = is_mask_background
        is_selected_B = get_selected(is_mask_background, num_selected_ab)
        is_mask_Bp = is_mask_background_bp
        is_selected_Bp = get_selected(is_mask_background_bp, num_selected_ab)
        is_mask_AB = is_mask_A & repeat(is_mask_B, 'B N -> (R B) N', R=R)
        is_selected_AB = get_selected(is_mask_AB, N - self.mask_m + self.b_max)
        is_patch_target = is_bp

        return is_selected_A, is_mask_A, is_selected_B, is_mask_B, is_selected_Bp, is_mask_Bp, is_selected_AB, is_mask_AB, is_patch_target, is_valid_target


    def get_mask(self, strategy, B_IN, N, device, R=1):
        if strategy == 'random':
            return self.sample_mask_random(B_IN, N, device, R=R)
        if strategy == 'block':
            return self.sample_mask_block(B_IN, N, device, R=R)
        raise ValueError(f'Unknown mask strategy: {strategy}')

    @torch.compile
    def forward_mar(self, x, x_target=None):
        # Merging the first iteration with different targets
        assert self.in_context_len > 0

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
        d = self.hidden_size

        R=self.batch_repeat_mar
        O=self.diff_repeat

        is_selected_A, is_mask_A, is_selected_B, is_mask_B, is_selected_Bp, is_mask_Bp, is_selected_AB, is_mask_AB, is_patch_target, is_valid_target = self.get_mask(self.mask_strategy_mar, B_IN, N, device, R=R)
        is_mask_encoder_AB = rearrange(is_mask_AB[is_selected_AB], '(X m) -> X m', X=R*B_IN)

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
        
        z_AB_factor_pos = get_factor_pos(is_mask_AB)

        z_t1_A_nopos = repeat(z_t1_og_nopos, 'B N d -> (R B) N d', R=R)

        z_t1_AB = torch.where(
            repeat(is_mask_B, 'B N -> (R B) N 1', R=R),
            z_t1_A_nopos,
            repeat(z_t1_og_nopos, 'B N d -> (R B) N d', R=R),
        ) + z_AB_factor_pos
                
        def get_encoded(z, is_selected, is_mask_encoder):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, d)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic)
            for block_idx, block in enumerate(self.encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
        
        z_e_AB_t1 = get_encoded(z_t1_AB, is_selected_AB, is_mask_encoder_AB)

        z_d_AB_t1_nomaskpos = z_e_AB_t1.new_zeros((R*B_IN,N,d)).masked_scatter(
            repeat(is_selected_AB, 'B N -> B N d', d=d),
            z_e_AB_t1[:, self.in_context_len:].flatten(),
        )
        z_d = torch.cat([
            z_e_AB_t1[:, :self.in_context_len],
            torch.where(
                repeat(is_mask_AB, 'RB N -> RB N 1'),
                repeat(self.mask_token.to(dtype), '1 1 d -> (R B) N d', R=R, B=B_IN, N=N),
                z_d_AB_t1_nomaskpos
            )
        ], dim=1)
        z_d = z_d + torch.cat([z_AB_factor_pos.new_zeros(R*B_IN, self.in_context_len, d), z_AB_factor_pos], dim=1)
        for block_idx, block in enumerate(self.decoder):
            z_d = block(z_d, self.feat_rope_incontext, attn_mask=None)

        # Section: Trains such that decoder used to merging multiple encoder inputs.
        # Tradeoff: This wastes compute by calculating B tokens during A mask. But necessary for constant masks
        # Masking is coupled here, but correct because ultimately is_mask_B valid check and is_mask_A masking combine to work
        z_m = repeat(z_d[:,self.in_context_len:][is_mask_A], 'Y d -> (O Y) d', O=O)
        feats_p_rb = repeat(x_p_target, 'B N d -> (R B) N d', R=R)
        feats_p_m = repeat(feats_p_rb[is_mask_A], 'Y d -> (O Y) d', O=O, d=feats_p_rb.size(-1))
        t_m = torch.where(torch.rand(z_m.size(0), 1, device=device) < 0.1, torch.rand(z_m.size(0), 1, device=device), torch.sigmoid(torch.randn(z_m.size(0), 1, device=device) * self.std_mar + self.mu_mar))
        eps_m = torch.randn_like(feats_p_m)
        feats_p_m_t = t_m * feats_p_m + (1-t_m) * eps_m
        feats_pred = self.diff_mlp(feats_p_m_t, t_m.flatten(), z_m)
        mse_is_valid = repeat(repeat(is_mask_B, 'B N -> (R B) N', R=R)[is_mask_A], 'Y -> (O Y) 1', O=O)
        mse_loss = 2*(F.huber_loss(feats_pred*mse_is_valid, feats_p_m*mse_is_valid, reduction='none', delta=10) / (1-t_m).clamp_min(0.33).pow(2)).mean() / mse_is_valid.float().mean()
        
        loss = mse_loss
        return loss

    @torch.compile
    def forward_repr(self, x, x_target=None):
        # Merging the first iteration with different targets
        # Goal: include Bp repeat
        assert self.in_context_len > 0


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
        d = self.hidden_size

        P=self.b_num_p
        R=self.batch_repeat
        R1=self.batch_repeat+1
        O=self.diff_repeat
        V=self.diff_repeat_repr

        is_selected_A, is_mask_A, is_selected_B, is_mask_B, is_selected_Bp, is_mask_Bp, is_selected_AB, is_mask_AB, is_patch_target, is_valid_target = self.get_mask(self.mask_strategy_repr, B_IN, N, device, R=R)
        is_selected_BABp = torch.cat([is_selected_B, is_selected_A, is_selected_Bp], 0)
        is_mask_BABp = torch.cat([is_mask_B, is_mask_A, is_mask_Bp], 0)
        is_mask_encoder_BABp = rearrange(is_mask_BABp[is_selected_BABp], '(X m) -> X m', X=(1+R+P)*B_IN)
        is_mask_encoder_AB = rearrange(is_mask_AB[is_selected_AB], '(X m) -> X m', X=R*B_IN)

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
        
        z_BABp_factor_pos, z_BABp_factor_pos_pad, z_BABp_encoder_rope_idx, z_BABp_factorpad_rope_idx, z_BABp_cls_rope_idx = get_factor_pos(is_mask_BABp)
        z_AB_factor_pos, z_AB_factor_pos_pad, z_AB_encoder_rope_idx, z_AB_factorpad_rope_idx, z_AB_cls_rope_idx = get_factor_pos(is_mask_AB)

        z_t1_A_nopos = repeat(z_t1_og_nopos, 'B N d -> (R B) N d', R=R)

        z_t1_BABp = torch.cat([
            z_t1_og_nopos,
            z_t1_A_nopos,
            torch.where(
                rearrange(is_patch_target, 'P B N -> (P B) N 1'), 
                repeat(z_target, "B N d -> (P B) N d", P=P) + rearrange(self.is_b_embed.to(dtype), '1 d -> 1 1 d'),
                repeat(z_t1_og_nopos, 'B N d -> (P B) N d', P=P),
            )
        ], dim=0) + z_BABp_factor_pos
        z_t1_AB = torch.where(
            repeat(is_mask_B, 'B N -> (R B) N 1', R=R),
            z_t1_A_nopos,
            repeat(z_t1_og_nopos, 'B N d -> (R B) N d', R=R),
        ) + z_AB_factor_pos
                

        def get_encoded(z, is_selected, is_mask_encoder):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, d)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic)
            for block_idx, block in enumerate(self.encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
        
        z_e_BABp_t1 = get_encoded(z_t1_BABp, is_selected_BABp, is_mask_encoder_BABp)
        z_e_AB_t1 = get_encoded(z_t1_AB, is_selected_AB, is_mask_encoder_AB)

        is_mask_BAB = torch.cat([is_mask_B, is_mask_AB], dim=0)
        z_d_B_t1_nomaskpos = z_e_BABp_t1.new_zeros((B_IN,N,d)).masked_scatter(
            repeat(is_selected_B, 'B N -> B N d', d=d),
            z_e_BABp_t1[:B_IN, self.in_context_len:].flatten(),
        )
        z_d_AB_t1_nomaskpos = z_e_AB_t1.new_zeros((R*B_IN,N,d)).masked_scatter(
            repeat(is_selected_AB, 'B N -> B N d', d=d),
            z_e_AB_t1[:, self.in_context_len:].flatten(),
        )
        z_d_BAB_t1_nomaskpos = torch.cat([z_d_B_t1_nomaskpos, z_d_AB_t1_nomaskpos], dim=0)
        z_d = torch.cat([
            torch.cat([z_e_BABp_t1[:B_IN, :self.in_context_len], z_e_AB_t1[:, :self.in_context_len]], dim=0),
            torch.where(
                repeat(is_mask_BAB, 'RB N -> RB N 1'),
                repeat(self.mask_token.to(dtype), '1 1 d -> (R B) N d', R=1+R, B=B_IN, N=N),
                z_d_BAB_t1_nomaskpos
            )
        ], dim=1)
        z_BAB_factor_pos = torch.cat([z_BABp_factor_pos[:B_IN], z_AB_factor_pos], dim=0)
        z_d = z_d + torch.cat([z_BAB_factor_pos.new_zeros((1+R)*B_IN, self.in_context_len, d), z_BAB_factor_pos], dim=1)
        for block_idx, block in enumerate(self.decoder):
            z_d = block(z_d, self.feat_rope_incontext, attn_mask=None)

        # Section: Trains such that decoder used to merging multiple encoder inputs.
        # Tradeoff: This wastes compute by calculating B tokens during A mask. But necessary for constant masks
        # Masking is coupled here, but correct because ultimately is_mask_B valid check and is_mask_A masking combine to work
        is_mask_BA = torch.cat([is_mask_B, is_mask_A], dim=0)
        z_m = repeat(z_d[:,self.in_context_len:][is_mask_BA], 'Y d -> (O Y) d', O=O)
        feats_p_rb = repeat(x_p_target, 'B N d -> (R B) N d', R=R1)
        feats_p_m = repeat(feats_p_rb[is_mask_BA], 'Y d -> (O Y) d', O=O, d=feats_p_rb.size(-1))
        t_m = torch.where(torch.rand(z_m.size(0), 1, device=device) < 0.1, torch.rand(z_m.size(0), 1, device=device), torch.sigmoid(torch.randn(z_m.size(0), 1, device=device) * self.std_mar + self.mu_mar))
        eps_m = torch.randn_like(feats_p_m)
        feats_p_m_t = t_m * feats_p_m + (1-t_m) * eps_m
        feats_pred = self.diff_mlp(feats_p_m_t, t_m.flatten(), z_m)
        mse_is_valid = repeat(repeat(is_mask_B, 'B N -> (R B) N', R=R1)[is_mask_BA], 'Y -> (O Y) 1', O=O)
        mse_loss = 2*(F.huber_loss(feats_pred*mse_is_valid, feats_p_m*mse_is_valid, reduction='none', delta=10) / (1-t_m).clamp_min(0.33).pow(2)).mean() / mse_is_valid.float().mean()
        
        with torch.no_grad():
            z_decoder_r = rearrange(z_d[:,self.in_context_len:], '(R B) N d -> R B N d', R=R1)
            z_bp_rp = z_decoder_r[:, torch.arange(B_IN, device=device), is_patch_target.int().argmax(-1)].float()
            z_bp_vpr = repeat(z_bp_rp, 'R P B d -> (V P R B) d', V=V)

            x_bp_vpr = repeat(repeat(x_p_target, 'B N d -> P B N d', P=P)[is_patch_target], '(P B) d -> (V P R B) d', P=P, R=R1, V=V).float()
            t_bp_vpr = torch.sigmoid(torch.randn(V,P,B_IN, device=device) * self.std_repr + self.mu_repr)
            t_bp_vpr = repeat(t_bp_vpr, 'V P B -> (V P R B) 1', R=R1).float()
            eps_bp_vpr = repeat(torch.randn(V,P,B_IN,x_bp_vpr.size(-1),device=device), 'V P B d -> (V P R B) d', R=R1).float()
            x_bp_vpr_noisy = t_bp_vpr * x_bp_vpr + (1-t_bp_vpr) * eps_bp_vpr

            x_bp_vpr_pred = self.diff_mlp(x_bp_vpr_noisy.to(dtype), t_bp_vpr.flatten().to(dtype), z_bp_vpr.to(dtype)).float()

            freq_weight = 1/2*t_bp_vpr/(1-t_bp_vpr).clamp_min(0.33).pow(2)
            h_p_vprb = freq_weight * (x_bp_vpr_pred - x_bp_vpr).pow(2).sum(-1, True)
            h_p_rbpv = rearrange(h_p_vprb, '(V P R B) 1 -> R B P V', P=P, R=R1, V=V).to(dtype)

        # REPR Calculation
        def get_repr_encoder(z_e, is_selected, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic)
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

        z_re_BABp_t1 = get_repr_encoder(z_e_BABp_t1, is_selected_BABp, is_mask_encoder_BABp)

        z_BABp_t1_repr = get_repr_decoder(z_re_BABp_t1, z_BABp_factor_pos_pad, z_BABp_factorpad_rope_idx, z_BABp_encoder_rope_idx[is_selected_BABp].view((1+R+P)*B_IN,-1), z_BABp_cls_rope_idx, is_mask_encoder_BABp)
        z_BA_t1_repr = rearrange(self.repr_final_layer(z_BABp_t1_repr[:(1+R)*B_IN]), '(R b) q d -> R b 1 q d', R=R1)
        z_Bp_t1_repr = rearrange(self.repr_final_layer(z_BABp_t1_repr[(1+R)*B_IN:]), '(P b) q d -> b P q d', P=P)
        z_t1_repr_global = rearrange(self.repr_final_layer_global(z_re_BABp_t1[:(1+R)*B_IN, :self.in_context_len]), '(R b) C d -> R b 1 (C d)', R=R1)
        z_Bp_t1_repr_global = rearrange(self.repr_final_layer_global(z_re_BABp_t1[(1+R)*B_IN:, :self.in_context_len]), '(P b) C d -> b P (C d)', P=P)

        z_BA_t1_repr = torch.where(torch.rand(z_BA_t1_repr.size(0),device=device).view(-1,1,1,1,1) < self.pad_dropout, z_BA_t1_repr*self.pad_dropout_mask.to(dtype), z_BA_t1_repr)

        # MINI TODO REMOVE IS EMPTY
        is_empty_BA = is_mask_BA.all(-1)
        z_BA_t1_repr = torch.where(is_empty_BA.view(R1, B_IN, 1, 1, 1), 0, z_BA_t1_repr)
        z_t1_repr_global = torch.where(is_empty_BA.view(R1, B_IN, 1, 1), 0, z_t1_repr_global)
        z_B_t1_repr = z_BA_t1_repr[:1]
        z_A_t1_repr = z_BA_t1_repr[1:]
        z_B_t1_repr_global = z_t1_repr_global[:1]
        z_A_t1_repr_global = z_t1_repr_global[1:]

        is_valid_pmi = reduce(~(repeat(~is_valid_target, '(R b) n -> R b P n', R=R1, P=P) & repeat(is_patch_target, 'P b n -> R b P n', R=R1)), 'R b P n -> R b P', 'all')  # don't allow loss for: ~is_mask and is_bp
        is_valid_B = is_valid_pmi[:1]
        is_valid_A = is_valid_pmi[1:]
        is_valid = is_valid_B & is_valid_A

        h_p_given_B_preintegration = h_p_rbpv[:1]
        h_p_given_BA_preintegration = h_p_rbpv[1:]

        pmi_pred = torch.einsum('rbqd,bpqd->rbp', z_A_t1_repr[:,:,0], z_Bp_t1_repr - z_B_t1_repr[0]) / (2 * z_A_t1_repr.size(-1) * z_A_t1_repr.size(-2)**0.5) * d**0.5  # GIGA TODO USE REPR HERE!
        pmi_pred_global = torch.einsum('rbd,bpd->rbp', z_A_t1_repr_global[:,:,0], z_Bp_t1_repr_global - z_B_t1_repr_global[0]) / (2 * z_A_t1_repr_global.size(-1)**0.5)
        pmi_target = (h_p_given_B_preintegration - h_p_given_BA_preintegration).mean(-1)  # GIGA TODO THIS CLAMPING IS ONLY GOOD FOR PIXELS (BUT SANCHECKED)

        # Factorization has an optimum at zero, so huber should be fine, no? Only kind of due to E[PMI est error], but PMI est bias dominates regardless
        repr_loss = (2*F.huber_loss(pmi_pred, pmi_target, reduction='none', delta=self.repr_delta) * is_valid / reduce(is_valid.bfloat16(), 'R b P -> R b 1', 'mean').clamp_min(1e-5)).mean()
        repr_loss_global = (2*F.huber_loss(pmi_pred_global, pmi_target, reduction='none', delta=self.repr_delta) * is_valid / reduce(is_valid.bfloat16(), 'R b P -> R b 1', 'mean').clamp_min(1e-5)).mean()

        loss = mse_loss + self.repr_loss_weight * repr_loss + self.repr_loss_weight * repr_loss_global
        return loss, mse_loss.detach().clone(), repr_loss.detach().clone(), repr_loss_global.detach().clone()


    @torch.compile
    def forward_feats(self, x, num_mask_tokens=0, return_padded=True):
        assert self.in_context_len > 0

        B_IN, C, H, W = x.shape
        device = x.device
        x_p_og = self.patchify(x)
        z_t1_og_nopos = self.x_embedder(x_p_og)
        dtype = z_t1_og_nopos.dtype

        N = self.num_patches
        d = self.hidden_size
        R=1

        # Preserve the zero-mask path's random-number order during training.
        mask_rank = torch.rand((B_IN, N), device=device).argsort(dim=1).argsort(dim=1)
        is_mask_rb = mask_rank < num_mask_tokens
        is_selected_rb = ~is_mask_rb
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

        z_t1 = repeat(z_t1_og_nopos, 'B N d -> (R B) N d', R=R) + z_factor_pos
                
        def get_encoded(z, is_selected, is_mask_encoder):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, d)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic)
            for block_idx, block in enumerate(self.encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
        
        z_e_t1 = get_encoded(z_t1, is_selected_rb, is_mask_encoder_rb)

        def get_repr_encoder(z_e, is_selected, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic)
            for block_idx, block in enumerate(self.repr_encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e

        z_re_t1 = get_repr_encoder(z_e_t1, is_selected_rb, is_mask_encoder_rb)

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

REPR_MODELS = {
    'REPR-XT/16': REPR_XT_16,
    'REPR-T/16': REPR_T_16,
    'REPR-S/16': REPR_S_16,
    'REPR-B/16': REPR_B_16,
    'REPR-L/16': REPR_L_16,
}
