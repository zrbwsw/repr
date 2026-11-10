# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
# 中文阅读路线：基础注意力/MLP → REPR 模块定义 → mask 采样 → forward_mar → forward_repr → forward_feats。
# Figure 1 中的 MAE 对应本文件的 self.encoder + self.decoder；diff_mlp 是之后独立接收噪声目标的预测头。
# 同一个 encoder 同时服务生成目标和表示学习。B 的编码会复用；A、B、Bp、AB 是同图的不同可见集合。
# 下面的中文说明只解释当前实现，保留原有代码、英文注释和 TODO；公式中的 t 采用本代码的方向。
# 核心约定：is_mask=True 表示内容被遮挡；attention 的布尔 mask=True 则表示允许注意，二者不要混淆。
from functools import partial

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from einops import rearrange, repeat, reduce


# μP 的参考宽度是 REPR-B 的 768；实际学习率分组见 get_mup_param_groups，而非改变 Linear 的前向公式。
MUP_BASE_WIDTH = 768


# 仅用子类类型标记需要宽度相关学习率缩放的矩阵；本类不覆盖 nn.Linear.forward。
class MuPLinear(nn.Linear):
    """Marks a hidden or output linear weight for MuP LR scaling."""


# RMSNorm 只按最后一维的均方根归一化，不像 LayerNorm 那样先减去均值。
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 归一化统计在 float32 中计算，再转回输入精度，减小混合精度下的数值误差。
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# 先把非拼接维广播到兼容大小，再沿 dim 拼接；这里主要用于合成二维 RoPE 的行、列频率。
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


# 将每一对相邻通道 (x1, x2) 变成 (-x2, x1)，配合 cos/sin 实现二维子空间旋转。
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


# 二维视觉 RoPE：前、后两部分通道分别携带行坐标和列坐标的旋转相位。
# 它修改 attention 的 Q/K，不等同于后面直接加在 token 上的可见中心相对位置 embedding。
class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        theta = 10000,
        num_cls_token = 0
    ):
        super().__init__()
        # 预先构造整张 pt_seq_len × pt_seq_len 网格的相位表；使用时按实际保留的 patch 索引取子集。
        freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = torch.arange(pt_seq_len) / pt_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)
        
        # 频率表是非持久 buffer：随模型迁移设备，本身不可学习，也不写入 state_dict。
        self.num_cls_token = num_cls_token
        self.register_buffer('freqs_cos', freqs.cos().view(-1, freqs.shape[-1]), persistent=False)
        self.register_buffer('freqs_sin', freqs.sin().view(-1, freqs.shape[-1]), persistent=False)

    def forward(self, t, m=None, use_cls=True, idx=None, m_selected=None, cls_idx=None):
        # t 在本方法里是 Q/K 张量 [batch, heads, tokens, head_dim]，不是扩散时间。
        # m 为网格上被排除的位置；idx 可直接给出坐标索引；m_selected 标识保留序列中的隔离补齐项。
        B, H, N, D = t.shape
        # two changes
        # 1: Subindex for m
        # 2: Legacy CLS padding reuses a selected token position unless cls_idx is given
        # 3: idx: B, N
        # 4: m_selected: B, N_selected; for CLS, copy the first unmasked selected token
        # 5: cls_idx: B; for v14, put CLS at the visible-image center

        # 显式 idx 的优先级高于 m；表示头 cross-attention 使用它索引带外圈 padding 的网格。
        if idx is not None:
            freqs_cos = self.freqs_cos.expand(B, -1, -1).gather(1, idx.unsqueeze(-1).expand(-1,-1,D)).view(B, 1, -1, D)
            freqs_sin = self.freqs_sin.expand(B, -1, -1).gather(1, idx.unsqueeze(-1).expand(-1,-1,D)).view(B, 1, -1, D)
        elif m is not None:
            freqs_cos = self.freqs_cos.expand(B, -1, -1)[~m].view(B, 1, -1, D)
            freqs_sin = self.freqs_sin.expand(B, -1, -1)[~m].view(B, 1, -1, D)
        else:
            freqs_cos = self.freqs_cos.view(1, 1, -1, D).expand(B, 1, -1, D)
            freqs_sin = self.freqs_sin.view(1, 1, -1, D).expand(B, 1, -1, D)

        # in-context/CLS token 的旋转坐标由调用方决定，并非任何地方都放在可见中心。
        # 给出 cls_idx 时用指定坐标；否则复用第一个位置，或第一个未遮挡的 selected token 的位置。
        # 若 selected 序列全部被遮挡，argmin 返回首项；这只是当前空视图分支的回退行为。
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

        # 相位只作用于 Q/K 的最后一维，不在这里修改 V，也不改变 token 的顺序。
        return  t * freqs_cos + rotate_half(t) * freqs_sin


# 把连续扩散时间映射到模型宽度：固定正弦/余弦特征之后接两层可学习 MLP。
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
        # 允许非整数时间；奇数维时末尾补零，使输出通道数恰为 dim。
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

    # 此处 t 才是每个噪声样本的标量时间；前向输出供 diffusion MLP 的 AdaLN 使用。
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# 统一封装 PyTorch SDPA。此处不施加因果约束，生成条件由可见集合和显式 attention mask 决定。
# 传给 SDPA 的 bool mask 中 True 表示允许访问，因此不能直接把 is_mask 当作它传入。
def scaled_dot_product_attention(query, key, value, dropout_p=0.0, attn_mask=None) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query, 
        key, 
        value, 
        attn_mask=attn_mask, 
        dropout_p=dropout_p, 
        is_causal=False
    )


# 自注意力：同一序列产生 Q/K/V，Q/K 分头做 RMSNorm 并施加 RoPE，再合并各头。
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

    # 外部提供 rope callable，因而同一注意力模块可处理完整网格、选中子序列和不同 CLS 坐标。
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


# 交叉注意力：Q 来自空间查询 x，K/V 来自已编码的可见上下文 context。
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

    # 查询与上下文的 token 数可不同，分别用 q_rope、k_rope 对齐其实际空间坐标。
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

# SwiGLU 前馈层：一路经 SiLU 作为门，乘上另一路，再投影回原宽度。
class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        # 乘 2/3 是对门控双投影的中间宽度调整；不改变外部 token 维度。
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = MuPLinear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = MuPLinear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


# MAE encoder、MAE decoder 和表示头前半段共用这种 block 结构，但各自拥有独立 block 参数。
# 其中 self.encoder 这一组参数在不同视图和不同损失之间共享。
class MAGEBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    # 两条残差支路依次是带 RoPE 的 self-attention 与逐 token 的 SwiGLU。
    @torch.compile(dynamic=False)
    def forward(self, x, feat_rope=None, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope=feat_rope, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# 表示头后半段的 cross-attention block；本类本身没有 query 之间的 self-attention。
# 每个位置查询反复读取同一份编码上下文，再经逐位置前馈层得到空间表示。
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

    # x 的残差保留位置查询，context 只提供 K/V；同一层内不会把 context 更新后写回。
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


# AdaLN 的仿射调制：scale=0、shift=0 时保持归一化结果；残差门控在调用方另算。
def modulate_1d(x, shift, scale):
    return x * (1 + scale) + shift

# diffusion MLP 的输出头：用时间与上下文条件调制 LayerNorm，再回归干净目标的各通道。
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

    # c 是已融合的条件向量，产生 shift/scale；输出维度由目标空间决定（像素或上一轮特征）。
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate_1d(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# 表示输出头也用 LayerNorm + shift/scale，但 shift/scale 是共享可学习参数，未由扩散时间生成。
# 空间表示和全局表示各有独立的这一输出头。
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

    # 保留前面的 batch/token 维，只把最后一维投影为 repr_dim。
    def forward(self, x):
        x = modulate_1d(self.norm_final(x), self.shift, self.scale)
        x = self.linear(x)
        return x

# diffusion MLP 的条件残差块；当前实现保持通道数不变，以 AdaLN 的第三组输出控制残差强度。
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

    # y 同时含扩散时间与 MAE 条件；门 gate_mlp 为零时，此块初始等价于恒等映射。
    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate_1d(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


# 每个目标 patch 独立调用的条件 diffusion MLP；它没有 patch 间 self-attention。
# 跨 patch 的可见上下文已由 MAE 的 E+D 聚合到 c；带噪目标 x 和时间 t 在这里才进入预测器。
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

        # 时间与条件分别投影到 model_channels；噪声输入由另一条 input_proj 支路进入。
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

    # 初始化区分外部输入投影、内部矩阵和 AdaLN；REPR 总初始化后还会重新调用本方法恢复这些设置。
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

        # 时间 MLP 后一矩阵的标准差随宽度缩放，参考宽度为 MUP_BASE_WIDTH。
        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(
            self.time_embed.mlp[2].weight,
            std=0.02 * math.sqrt(MUP_BASE_WIDTH / self.model_channels),
        )

        # 将条件调制和最终输出投影置零，使残差块从恒等映射、整个预测头从零输出起步。
        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # 三项输入 x/t/c 分工固定；这里的 forward 返回干净目标预测，训练调用处直接与原目标比较。
    @torch.compile
    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        # x: [目标样本数, target_dim]；t: [目标样本数]；c: [目标样本数, hidden_size]。
        # 这里的样本数通常已把 batch、位置和噪声重复维展平，不等于原始图像 batch size。
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        # 相加融合时间与上下文，随后每个残差块用 y 生成 AdaLN 参数；不是把 B/AB 原图直接拼给 MLP。
        y = t + c

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)


# RePR 主体。建议先读 forward_repr 的阶段注释，再回看掩码、位置编码和基础模块。
# 生成路线：可见图像 patch → encoder → 补 mask token → decoder → 目标位置条件 → diff_mlp。
# 表示路线：同一 encoder → repr_encoder 的 3 层 self-attention → 3 层 cross-attention → 表示输出头。
# forward_mar 只训练生成目标；forward_repr 联合训练生成目标、空间 PMI 表示和全局 PMI 表示。
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
        p_repeat=8,
        diff_repeat_repr=32,
        b_max_visible=48,
        a_min_mask=256 - 76,
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
        # 方法签名给出类的默认值；论文/README 的训练命令会覆盖其中部分值，不能把二者混为一谈。
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        # 像素 patch_dim = C × patch_size²；256×256、patch=16 时 N=256，RGB 目标为 768 维。
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.in_context_len = in_context_len
        self.repr_dim = repr_dim
        self.batch_repeat = batch_repeat
        self.batch_repeat_mar = batch_repeat_mar
        self.diff_repeat = diff_repeat
        self.a_min_mask = a_min_mask
        self.mask_strategy_mar = mask_strategy_mar
        self.mask_strategy_repr = mask_strategy_repr
        self.repr_loss_weight = repr_loss_weight
        self.repr_delta = repr_delta
        self.pad_dropout = pad_dropout
        self.is_first_iter = is_first_iter
        self.mu_repr, self.std_repr, self.mu_mar, self.std_mar = mu_repr, std_repr, mu_mar, std_mar

        # 所有可见上下文首先来自当前图像的像素 patch；第二轮也不会整体替换成教师特征输入。
        # linear embed
        self.x_embedder = nn.Linear(self.patch_dim, hidden_size, bias=True)

        # 可学习 in-context tokens 相当于多枚全局上下文槽；不是类别标签，也不是某个原始 patch。
        if self.in_context_len > 0:
            self.in_context_tokens = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_tokens, std=.02)

        # 这张 RoPE 表面向原始 patch 网格；表示头的扩展网格 RoPE 在后面单独建立。
        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # Figure 1 的共享 E：四种可见集合分别运行这同一组 encoder；预设型号均使用 18 层。
        # transformer
        self.encoder = nn.ModuleList([
            MAGEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for i in range(depth)
        ])

        # Figure 1 的 D：预设为 6 层，输出条件特征而非直接输出像素；像素/特征预测交给 diff_mlp。
        self.decoder = nn.ModuleList([
            MAGEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for i in range(decoder_depth)
        ])
        # 只有恢复完整网格给 decoder 时，缺失位置才统一替换成这个可学习 mask token。
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # P 控制每图选择多少个单 patch 增量；V=diff_repeat_repr 控制每个增量的 Monte Carlo 噪声样本数。
        self.p_repeat = p_repeat
        self.b_max_visible = b_max_visible
        self.hw_len = input_size // patch_size
        self.diff_repeat_repr = diff_repeat_repr

        # 表示查询在原图四周各扩展一圈位置：16×16 对应 18×18=324 个空间查询。
        # 这扩展的是表示查询坐标，不是在原始输入图像中读取额外像素。
        self.factor_pad = 1
        self.pad_hw = self.hw_len + 2*self.factor_pad
        pad_dropout_mask = torch.zeros(self.pad_hw, self.pad_hw)
        pad_dropout_mask[self.factor_pad:self.factor_pad+self.hw_len,self.factor_pad:self.factor_pad+self.hw_len] = 1
        self.register_buffer('pad_dropout_mask', pad_dropout_mask.flatten()[...,None])
        # 可学习加性位置表按“当前位置相对可见中心的位移”查询；该位移表大小不同于查询网格大小。
        self.factor_posembed = nn.Parameter(torch.zeros((hw_seq_len*2-1+2*self.factor_pad)**2, hidden_size), requires_grad=True)
        torch.nn.init.normal_(self.factor_posembed, std=.02)
        self.feat_rope_incontext_pad = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len + 2*self.factor_pad,
            num_cls_token=self.in_context_len
        )
        # 空间输出按每个查询给出 repr_dim；全局输出对每枚 in-context token 投影后再展平。
        self.repr_final_layer = FinalLayerLearnedModulation(hidden_size, repr_dim)
        self.repr_final_layer_global = FinalLayerLearnedModulation(hidden_size, repr_dim)
        # 此变量名叫 representation_encoder，实际装的是 3 个 cross-attention block，承担表示头的读取阶段。
        self.representation_encoder = nn.ModuleList([
            RepresentationCrossAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(3)
        ])


        # 增量身份标记：后面仅加到 Bp 中新增的目标 p 上；名字 is_b_embed 不表示给整个 B 加标记。
        self.is_b_embed = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        torch.nn.init.normal_(self.is_b_embed, std=.02)

        # 表示头先对 encoder 序列再做 3 层 self-attention，再进入上面的空间查询 cross-attention。
        self.repr_encoder = nn.ModuleList([
            MAGEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for i in range(3)
        ])

        # 第一轮预测原始像素 patch；后续轮预测被冻结的上一轮模型给出的 repr_dim 维空间特征。
        # feats_embedder 只把这种目标特征映射为 Bp 新增位置的输入 embedding，不替换 A/B 的其他像素上下文。
        if self.is_first_iter:
            self.diff_mlp = SimpleMLPAdaLN(self.patch_dim, hidden_size, self.patch_dim, hidden_size, diff_mlp_depth)
        else:
            self.feats_embedder = nn.Linear(self.repr_dim, hidden_size, bias=True)
            self.diff_mlp = SimpleMLPAdaLN(self.repr_dim, hidden_size, self.repr_dim, hidden_size, diff_mlp_depth)

        self.initialize_weights()

    # 兼容旧 checkpoint 的键名：迁移增量标记并移除旧 noise_bin_limits；此处不改变模型前向结构。
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        old_key = prefix + "is_d_embed"
        if old_key in state_dict:
            state_dict.setdefault(prefix + "is_b_embed", state_dict.pop(old_key))
        state_dict.pop(prefix + "noise_bin_limits", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # 先统一初始化线性层，再覆盖 patch 输入投影和 diffusion MLP 的专用初始化。
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

    # μP 参数分组只给 MuPLinear.weight 乘 768/hidden_size 的学习率倍率；普通 Linear 与 bias 不因此缩放。
    # 一维参数和以 .bias 结尾的参数关闭 weight decay，其他参数沿用传入值。
    # 返回 lr_scale 供外部优化器/学习率调度读取，本方法本身不执行优化器更新。
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


    # 把每张图按行优先顺序切成非重叠 patch；每个 patch 内按 C、局部行、局部列展平。
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
    
    # 输入 m=True 代表隔离补齐项；输出 True 则代表“同组，允许 attention”。
    # 可见 patch 与 in-context tokens 同属 False 组，只能互相读；补齐的 masked tokens 只读自己的 True 组。
    # 因此不能把这里误读为“masked token 仍能把隐藏内容泄漏给可见 token”。
    def get_attention_mask(self, m):
        k = m.unsqueeze(1)
        q = m.unsqueeze(2)
        return (q == k).unsqueeze(1)


    # 构造近似均匀覆盖 [min_masked, N] 的遮挡数量，并平衡整个 batch 的总数，再随机分配给样本。
    # 固定/平衡总量配合后面的定长 selected 序列，减少 torch.compile 遇到的动态形状。
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

    # 随机 patch 掩码策略。B_IN 是图像数，R 是 A 视图数，P 是每图尝试的新增 patch 数。
    # 返回值同时包含逻辑可见性 is_mask 与物理选中序列 is_selected；两者不能当成互补关系。
    def sample_mask_random(self, B_IN, N, device, H=16, W=16, R=1):
        assert H * W == N

        P = self.p_repeat
        b_max_visible = min(self.b_max_visible, N)

        # 先确定每图遮挡数量，再按独立随机排名选择位置；True 表示该位置不可作为可见上下文。
        def random_mask(B, min_masked):
            counts = self._balanced_mask_counts(B, N, min_masked, device)
            rank = torch.rand(B, N, device=device).argsort(dim=1).argsort(dim=1)
            return rank < counts[:, None]

        # 每张图有 R 个 A，但共享一个稀疏 B；A/B 可以重叠。B 最多看见 b_max_visible 个 patch。
        is_mask_A = random_mask(R * B_IN, self.a_min_mask)
        is_mask_B = random_mask(B_IN, N - b_max_visible)

        # 优先在所有 A 与 B 都遮住的位置中选 P 个 p；若共同候选不足，排序仍会补入其他位置。
        # 因此训练末尾仍必须用 is_valid 筛掉 p 已可见的配对，而不能假设这里绝对保证全部有效。
        is_valid_A = rearrange(is_mask_A, '(R B) N -> R B N', R=R).all(dim=0)
        is_valid_bp = is_valid_A & is_mask_B
        is_bp_idx = (torch.rand(B_IN, N, device=device) + is_valid_bp.float()).argsort(dim=1, descending=True)[:, :P]
        is_bp = torch.zeros(P * B_IN, N, device=device, dtype=bool).scatter(1, rearrange(is_bp_idx, 'B P -> (P B) 1'), True)
        is_bp = rearrange(is_bp, '(P B) N -> P B N', P=P)

        # Bp 只把对应 p 从 B 的遮挡集合中移除，即可见集合 B ∪ {p}。
        is_mask_Bp = repeat(is_mask_B, 'B N -> P B N', P=P) & (~is_bp)

        # 优先选中所有可见位置，再从被遮挡位置补足固定长度。
        # 这些额外项带原始 embedding，但后续 attention 隔离并丢弃/屏蔽其输出，避免信息泄漏。
        def get_selected(mask, num_selected):
            rank = torch.rand_like(mask.float()).masked_fill(~mask, -1).argsort(dim=1)
            return torch.zeros_like(mask).scatter(1, rank[:, :num_selected], True)

        # A/B/Bp 使用相同 selected 长度以便沿 batch 维拼接；AB 的可见数上界是 A 与 B 的上界之和。
        num_selected_ab = max(N - self.a_min_mask, self.b_max_visible + 1)
        is_selected_A = get_selected(is_mask_A, num_selected_ab)
        is_selected_B = get_selected(is_mask_B, num_selected_ab)
        is_selected_Bp = get_selected(rearrange(is_mask_Bp, 'P B N -> (P B) N'), num_selected_ab)
        # 逻辑遮挡取交集，对应可见集合取并集：AB = A ∪ B，而不是把两个图像相加。
        is_mask_AB = is_mask_A & repeat(is_mask_B, 'B N -> (R B) N', R=R)
        is_selected_AB = get_selected(is_mask_AB, N - self.a_min_mask + self.b_max_visible)
        # 保存 B、各 A 的遮挡信息，供联合步骤验证每个 p 是否仍是该条件下的未知目标。
        is_valid_target = torch.cat([is_mask_B, is_mask_A], dim=0)

        return is_selected_A, is_mask_A, is_selected_B, is_mask_B, is_selected_Bp, rearrange(is_mask_Bp, 'P B N -> (P B) N'), is_selected_AB, is_mask_AB, is_bp, is_valid_target

    # 默认 block 策略：A 的位置优先级带局部块状相关性；B 仍是稀疏随机可见集合。
    # 这里 block=4 作用于优先级场，不等于为每张图硬切出一个固定的 4×4 可见矩形。
    def sample_mask_block(self, B_IN, N, device, H=16, W=16, R=1):
        # Guarantees 
        # m_index 1 has min_mask_ratio selected
        # m_index 2 is uniform over [min_mask_ratio, 1] per batch index.
        #   Global sum equal for compile
        # is_mask_background_2: background without B, shape b,n
        # is_mask_background_2_bp: background without BP, exactly 1 fewer true per element, shape d,b,n
        # is_mask_background_1_bp: background without BP MAE encoder mask  TODO clever 2x speedup.
        
        # 内部 block_mask 仍先平衡每个样本的遮挡数量，再用局部相关的随机场决定具体位置。
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

            # 对随机网格做环形 padding 与 max-pool，使邻近位置的优先级相关；微小随机项用于打破并列。
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

        P = self.p_repeat

        # is_mask_encoder 即 R 个 A 的遮挡掩码；数量为 [R*B_IN, N]，每个 A 最少遮 a_min_mask 个位置。
        is_mask_encoder = block_mask(R*B_IN,N,self.a_min_mask)

        is_valid_mask = rearrange(is_mask_encoder, 'RB (H W) -> RB H W', H=H, W=W).clone()
        is_valid_target = is_valid_mask.clone()
        is_valid_target = rearrange(is_valid_target, 'RB H W -> RB (H W)')

        # 按跨 R 个 A 的“可见次数”从少到多挑 p，并用小随机数打破并列。
        # 这只是尽量让 p 对所有 A 有效；交集不足时仍可能选中某个 A 已看见的位置，末尾会过滤。
        is_bp_idx = (repeat(~is_valid_target, '(R B) N -> R B N', R=R).sum(dim=0) + torch.rand((B_IN,N), device=device) * 0.5).argsort(dim=1)[:,:P]
        is_bp = torch.zeros(P*B_IN, N, device=device, dtype=bool).scatter(1, rearrange(is_bp_idx, 'b p -> (p b) 1'), 1)
        is_bp = rearrange(is_bp, '(P B) N -> P B N', P=P)

        # 每张图的 B 可见数 k 在 0..b_max_visible 间采样，允许 B 为空。
        b_max_visible = min(self.b_max_visible, N)
        k = torch.randint(0, b_max_visible + 1, (B_IN, 1), device=device)  # how many False per row
        # is_bp.any(0) 将全部 P 个目标位置排到尾部；在 README 的 P=16、b_max_visible=16 配置下，B 能避开全部 p。
        is_background_idx = (torch.rand(B_IN, N, device=device) + is_bp.any(0)).argsort(dim=1)[:, :b_max_visible]  # 0% chance take is_bp
        is_mask_background = torch.ones(B_IN, N, dtype=torch.bool, device=device).scatter(1, is_background_idx, torch.arange(b_max_visible, device=device).unsqueeze(0) >= k)
        # 为每个 p 复制同一个 B 并只打开该位置，得到 [P, B_IN, N] 的 Bp 视图。
        is_mask_background_bp = repeat(is_mask_background, 'B N -> P B N', P=P) & (~is_bp)
        # 以下若干 background/selected 中间变量保留原实现；最终返回所用的 selected 掩码会在后面统一构造。
        is_selected_background_bp = torch.zeros(P, B_IN, N, dtype=bool, device=device).scatter(2, is_mask_background_bp.int().argsort(2)[:,:,:b_max_visible+1], True)
        
        is_selected_background = is_selected_background_bp[0]  # Any element works
        is_selected_background_bp = rearrange(is_selected_background_bp, 'P B N -> (P B) N')
        is_mask_background_bp = rearrange(is_mask_background_bp, 'P B N -> (P B) N')

        is_mask = torch.cat([is_mask_background,is_mask_encoder],dim=0)  # GIGA TODO REQUIRES IS B < IS M
        is_selected_encoder = torch.zeros_like(is_mask).scatter(1, torch.rand_like(is_mask.float()).masked_fill(~is_mask, -1).argsort(dim=1)[:, : N - self.a_min_mask], True)
        is_valid_target = torch.cat([is_mask_background, is_valid_target], dim=0)


        # 同随机策略一样：visible 全部选入，masked 部分只为补齐固定长度；不要把 selected=True 误读成可见。
        def get_selected(mask, num_selected):
            return torch.zeros_like(mask).scatter(1, torch.rand_like(mask.float()).masked_fill(~mask, -1).argsort(dim=1)[:, : num_selected], True)
        
        # 返回的视图顺序为 A、B、Bp、AB；每项同时返回 selected 与 mask。
        # is_patch_target 则单独保留 [P, B_IN, N] 的 one-hot 目标位置，不是新的图像视图。
        num_selected_ab = max(N-self.a_min_mask, self.b_max_visible+1)
        is_mask_A = is_mask_encoder
        is_selected_A = get_selected(is_mask_encoder, num_selected_ab)
        is_mask_B = is_mask_background
        is_selected_B = get_selected(is_mask_background, num_selected_ab)
        is_mask_Bp = is_mask_background_bp
        is_selected_Bp = get_selected(is_mask_background_bp, num_selected_ab)
        # AB 的未知位置必须同时不在 A 和 B 中；有效目标 p 的验证在 forward_repr 中执行。
        is_mask_AB = is_mask_A & repeat(is_mask_B, 'B N -> (R B) N', R=R)
        is_selected_AB = get_selected(is_mask_AB, N - self.a_min_mask + self.b_max_visible)
        is_patch_target = is_bp

        return is_selected_A, is_mask_A, is_selected_B, is_mask_B, is_selected_Bp, is_mask_Bp, is_selected_AB, is_mask_AB, is_patch_target, is_valid_target


    # 分发采样策略。此调用沿用采样器 H=W=16 的默认网格，匹配 README 的 256/16 配置。
    def get_mask(self, strategy, B_IN, N, device, R=1):
        if strategy == 'random':
            return self.sample_mask_random(B_IN, N, device, R=R)
        if strategy == 'block':
            return self.sample_mask_block(B_IN, N, device, R=R)
        raise ValueError(f'Unknown mask strategy: {strategy}')

    # 生成训练步骤：只构造 AB 条件，不计算表示头或 PMI 监督。
    # 常见调度中 4 步里有 3 步走这里；联合步骤的 MAR 部分还会同时训练 B 条件。
    @torch.compile
    def forward_mar(self, x, x_target=None):
        assert self.in_context_len > 0

        B_IN, C, H, W = x.shape
        device = x.device
        # x_p_og 保存干净像素 patch 目标；z_og_nopos 是供共享 encoder 使用的像素输入 embedding。
        x_p_og = self.patchify(x)
        z_og_nopos = self.x_embedder(x_p_og)
        dtype = z_og_nopos.dtype

        # 第二轮可以传入上一轮冻结模型的特征作为 x_target；当前 encoder 的普通上下文仍来自 x。
        if x_target is None:
            x_p_target = x_p_og
        else:
            x_p_target = x_target

        N = self.num_patches
        d = self.hidden_size

        # R 是当前 MAR 的 A/AB 视图重复数，O 是每个目标位置的 diffusion 训练重复数。
        R=self.batch_repeat_mar
        O=self.diff_repeat

        # 这里虽取回完整 mask 元组，实际只编码 AB；is_mask_encoder_AB 标记 selected 序列中的隔离补齐项。
        is_selected_A, is_mask_A, is_selected_B, is_mask_B, is_selected_Bp, is_mask_Bp, is_selected_AB, is_mask_AB, is_patch_target, is_valid_target = self.get_mask(self.mask_strategy_mar, B_IN, N, device, R=R)
        is_mask_encoder_AB = rearrange(is_mask_AB[is_selected_AB], '(X m) -> X m', X=R*B_IN)

        # 按当前可见 patch 的二维坐标计算向下取整的中心，再查相对位移的加性位置 embedding。
        # 可见集合为空时分母 clamp 为 1，中心回退到 (0, 0)；这与 RoPE 的坐标选择是两件事。
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

        z_A_nopos = repeat(z_og_nopos, 'B N d -> (R B) N d', R=R)

        # A/B 都来自同一张原图，真正控制可见性的是后续 mask；torch.where 并非把 AB 内容交给 MLP。
        z_AB = torch.where(
            repeat(is_mask_B, 'B N -> (R B) N 1', R=R),
            z_A_nopos,
            repeat(z_og_nopos, 'B N d -> (R B) N d', R=R),
        ) + z_AB_factor_pos
                
        # 编码固定长度的 selected 子序列，并在前面插入可学习 in-context tokens。
        def get_encoded(z, is_selected, is_mask_encoder):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, d)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            # in-context 标为 False，和真实可见 patch 同组；masked 补齐项不能通过 self-attention 影响这一组。
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic)
            # encoder RoPE 用原网格 selected 坐标；此调用的 CLS 坐标取首个可见 selected token，而非显式可见中心。
            for block_idx, block in enumerate(self.encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
        
        z_e_AB = get_encoded(z_AB, is_selected_AB, is_mask_encoder_AB)

        # 把 selected 编码散射回完整 N 个空间位置；下一步将 masked 补齐项的内容全部替换掉。
        z_d_AB_nomaskpos = z_e_AB.new_zeros((R*B_IN,N,d)).masked_scatter(
            repeat(is_selected_AB, 'B N -> B N d', d=d),
            z_e_AB[:, self.in_context_len:].flatten(),
        )
        # Decoder 输入 = in-context 编码 + 完整 patch 网格，其中不可见位置用统一 mask_token 占位。
        z_d = torch.cat([
            z_e_AB[:, :self.in_context_len],
            torch.where(
                repeat(is_mask_AB, 'RB N -> RB N 1'),
                repeat(self.mask_token.to(dtype), '1 1 d -> (R B) N d', R=R, B=B_IN, N=N),
                z_d_AB_nomaskpos
            )
        ], dim=1)
        # decoder 再加可见中心相对位置 embedding；下面的 RoPE 是完整原网格，未传 cls_idx。
        z_d = z_d + torch.cat([z_AB_factor_pos.new_zeros(R*B_IN, self.in_context_len, d), z_AB_factor_pos], dim=1)
        for block_idx, block in enumerate(self.decoder):
            z_d = block(z_d, self.feat_rope_incontext, attn_mask=None)

        # 为保持可编译的目标总数，先在 A 遮挡位置取条件，再用 is_mask_B 筛出真正 AB 未见的位置。
        # 因此有些 B 已看见的位置会被多算一次，但其损失被置零，不会形成“复制已知目标”的训练项。
        z_m = repeat(z_d[:,self.in_context_len:][is_mask_A], 'Y d -> (O Y) d', O=O)
        feats_p_rb = repeat(x_p_target, 'B N d -> (R B) N d', R=R)
        feats_p_m = repeat(feats_p_rb[is_mask_A], 'Y d -> (O Y) d', O=O, d=feats_p_rb.size(-1))
        # MAR 时间分布混合 10% 均匀样本与 90% logit-normal 样本；每个目标还重复 O 次。
        t_m = torch.where(torch.rand(z_m.size(0), 1, device=device) < 0.1, torch.rand(z_m.size(0), 1, device=device), torch.sigmoid(torch.randn(z_m.size(0), 1, device=device) * self.std_mar + self.mu_mar))
        eps_m = torch.randn_like(feats_p_m)
        # 代码方向：x_t = t*x + (1-t)*eps，t=0 为纯噪声，t=1 为干净目标；与某些论文伪码的 t 方向相反。
        feats_p_m_t = t_m * feats_p_m + (1-t_m) * eps_m
        # MAE 只提供 z_m 条件；带噪目标与时间是 diffusion MLP 的另外两个独立输入。
        feats_pred = self.diff_mlp(feats_p_m_t, t_m.flatten(), z_m)
        mse_is_valid = repeat(repeat(is_mask_B, 'B N -> (R B) N', R=R)[is_mask_A], 'Y -> (O Y) 1', O=O)
        # 实际为时间加权的 2*Huber(delta=10)，小残差时等于平方误差；(1-t) 下限 0.33 限制高权重。
        # 乘 mask 后再除有效占比，使只对 AB 真正未知的位置求平均；mse_loss 是沿用的变量名。
        mse_loss = 2*(F.huber_loss(feats_pred*mse_is_valid, feats_p_m*mse_is_valid, reduction='none', delta=10) / (1-t_m).clamp_min(0.33).pow(2)).mean() / mse_is_valid.float().mean()
        
        loss = mse_loss
        return loss

    # 联合 RePR 步骤阅读路线：①目标与 mask → ②位置坐标 → ③共享 E → ④B/AB 的 D → ⑤MAR。
    # 随后 ⑥无梯度 PMI 估计 → ⑦A/B/Bp 表示 → ⑧空间与全局 PMI 回归；最后把三项损失相加。
    @torch.compile
    def forward_repr(self, x, x_target=None):
        # Merging the first iteration with different targets
        # Goal: include Bp repeat
        assert self.in_context_len > 0


        B_IN, C, H, W = x.shape
        device = x.device
        x_p_og = self.patchify(x)
        z_og_nopos = self.x_embedder(x_p_og)
        dtype = z_og_nopos.dtype

        # ① 确定目标空间。第一轮 target 是像素；第二轮 x_target 是外部冻结教师产生的特征。
        # 只有 Bp 新增的 p 需要 z_target：第一轮用像素 embedding，第二轮用 feats_embedder 特征投影。
        if x_target is None:
            x_p_target = x_p_og
            z_target = z_og_nopos
        else:
            x_p_target = x_target
            z_target = self.feats_embedder(x_target)

        N = self.num_patches
        d = self.hidden_size

        # 维度字典：B_IN=原图 batch；N=patch 数；d=backbone 宽度；P=每图新增 patch 数。
        # R=每图 A 视图数；R1=1+R，首项 B、其余 AB；O=MAR 噪声重复；V=PMI Monte Carlo 重复。
        # 这些 repeat 都是构造同图不同条件/噪声的训练样本，不表示顺序运行 V 步去噪采样。
        P=self.p_repeat             # P Repeat
        R=self.batch_repeat         # A Repeat
        R1=self.batch_repeat+1      # A Repeat + 1 (AB and B views concatenated)
        O=self.diff_repeat          # MAR diffloss repeat (From MAR Paper)
        V=self.diff_repeat_repr     # Monte Carlo Integration Repeat

        # 逻辑掩码说明：A/B/Bp/AB 的 True 均表示不可见；is_patch_target 中 True 则仅标记目标 p 的坐标。
        is_selected_A, is_mask_A, is_selected_B, is_mask_B, is_selected_Bp, is_mask_Bp, is_selected_AB, is_mask_AB, is_patch_target, is_valid_target = self.get_mask(self.mask_strategy_repr, B_IN, N, device, R=R)
        # 在 batch 维拼接 [B, A_1..A_R, Bp_1..Bp_P]，各视图之间不会互相 attention。
        # 对应 batch 大小为 (1+R+P)*B_IN；AB 单独编码，大小为 R*B_IN。
        is_selected_BABp = torch.cat([is_selected_B, is_selected_A, is_selected_Bp], 0)
        is_mask_BABp = torch.cat([is_mask_B, is_mask_A, is_mask_Bp], 0)
        is_mask_encoder_BABp = rearrange(is_mask_BABp[is_selected_BABp], '(X m) -> X m', X=(1+R+P)*B_IN)
        is_mask_encoder_AB = rearrange(is_mask_AB[is_selected_AB], '(X m) -> X m', X=R*B_IN)

        # ② 同时准备加性中心相对位置与 RoPE 索引。两套机制并存，不能一概称为“所有位置都减去中心”。
        # Center of Visible Tokens Position Embeddings
        def get_factor_pos(is_mask):
            device = is_mask.device
            B_ = is_mask.size(0)
            is_mask_rc = rearrange(is_mask, 'b (h w) -> b h w', h=self.hw_len)
            pos_h = repeat(torch.arange(self.hw_len, device=device), 'h -> b h w', b=B_, w=self.hw_len)
            pos_w = repeat(torch.arange(self.hw_len, device=device), 'w -> b h w', b=B_, h=self.hw_len)
            # 中心由真实可见位置计算，补齐的 masked tokens 不参与；行、列均使用整数向下取整。
            mean_pos_h = ((pos_h*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            mean_pos_w = ((pos_w*(~is_mask_rc)).sum((1,2))//(~is_mask_rc).sum((1,2)).clamp_min(1)).clamp(0,self.hw_len - 1)
            offset_pos_h = rearrange(pos_h, 'b h w -> b (h w)') - mean_pos_h.unsqueeze(-1)
            offset_pos_w = rearrange(pos_w, 'b h w -> b (h w)') - mean_pos_w.unsqueeze(-1)
            # factor_pos 直接加到内容 embedding 上，索引的是以可见中心为参照的位移表。
            offset_idx = (offset_pos_h + self.hw_len - 1 + self.factor_pad) * (2 * self.hw_len - 1 + 2 * self.factor_pad) + (offset_pos_w + self.hw_len - 1 + self.factor_pad)
            factor_pos = F.embedding(offset_idx, self.factor_posembed.to(dtype))
            
            # factor_pos_pad 为包括外圈在内的全部空间查询建立同样的中心相对位移 embedding。
            pad_pos_h = repeat(torch.arange(self.pad_hw, device=device), 'h -> b h w', b=B_, w=self.pad_hw)
            pad_pos_w = repeat(torch.arange(self.pad_hw, device=device), 'w -> b h w', b=B_, h=self.pad_hw)
            pad_offset_idx = (pad_pos_h - mean_pos_h[:,None,None] + self.hw_len - 1) * (self.pad_hw + self.hw_len - 1) + (pad_pos_w - mean_pos_w[:,None,None] + self.hw_len - 1)
            factor_pos_pad = F.embedding(rearrange(pad_offset_idx, 'b h w -> b (h w)'), self.factor_posembed.to(dtype))

            # patch_rope_idx 把真实 patch 放到扩展网格内侧；pad_rope_idx 覆盖整张扩展查询网格。
            # cls_rope_idx 才显式给出可见中心，后面只在表示头 cross-attention 的 K 侧传入它。
            # 主 encoder/repr_encoder 与 MAE decoder 的调用方式不同，不能据此声称它们的 CLS RoPE 都位于中心。
            patch_rope_idx = rearrange((pos_h + self.factor_pad) * self.pad_hw + (pos_w + self.factor_pad), 'b h w -> b (h w)')
            pad_rope_idx = repeat(torch.arange(self.pad_hw * self.pad_hw, device=device), 'q -> b q', b=B_)
            cls_rope_idx = (mean_pos_h + self.factor_pad) * self.pad_hw + mean_pos_w + self.factor_pad

            return factor_pos, factor_pos_pad, patch_rope_idx, pad_rope_idx, cls_rope_idx
        
        z_BABp_factor_pos, z_BABp_factor_pos_pad, z_BABp_encoder_rope_idx, z_BABp_factorpad_rope_idx, z_BABp_cls_rope_idx = get_factor_pos(is_mask_BABp)
        z_AB_factor_pos, z_AB_factor_pos_pad, z_AB_encoder_rope_idx, z_AB_factorpad_rope_idx, z_AB_cls_rope_idx = get_factor_pos(is_mask_AB)

        # A 的像素 embedding 沿 R 复制；各 A 的差异由 mask 和对应的中心位置 embedding 决定。
        z_A_nopos = repeat(z_og_nopos, 'B N d -> (R B) N d', R=R)

        # B、A、Bp 按上面的 batch 顺序准备内容，其中 Bp 的新增位置需要特殊处理。
        z_BABp = torch.cat([
            z_og_nopos,
            z_A_nopos,
            # 只在 Bp 的目标 p 位置替换为 z_target 并加 is_b_embed，以标明哪一个是新加入的增量。
            # 其他 Bp 位置仍用原图像素 embedding；第二轮也不把所有 B 的上下文改成教师特征。
            torch.where(
                rearrange(is_patch_target, 'P B N -> (P B) N 1'), 
                repeat(z_target, "B N d -> (P B) N d", P=P) + rearrange(self.is_b_embed.to(dtype), '1 d -> 1 1 d'),
                repeat(z_og_nopos, 'B N d -> (P B) N d', P=P),
            )
        ], dim=0) + z_BABp_factor_pos
        # AB 使用 A∪B 的可见性与自己的中心位置；它不含 Bp 的增量身份标记。
        z_AB = torch.where(
            repeat(is_mask_B, 'B N -> (R B) N 1', R=R),
            z_A_nopos,
            repeat(z_og_nopos, 'B N d -> (R B) N d', R=R),
        ) + z_AB_factor_pos
                
        # ③ 所有视图调用同一 self.encoder。Figure 1 上方表示路线与下方 MAE 条件路线共享 E。
        # Encoder
        def get_encoded(z, is_selected, is_mask_encoder):
            B_ = z.size(0)
            # selected 是物理定长子序列，可能包含为补齐而选入的遮挡 patch；可见性还要看 is_mask_encoder。
            z_e = z[is_selected].view(B_, -1, d)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            # 把 in-context tokens 和可见 patch 归到同组，利用 attention mask 隔离补齐内容。
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic)
            # 此处 RoPE 不传 cls_idx，沿用首个可见 selected token 的坐标；加性位置 embedding 则已在外面按中心建立。
            for block_idx, block in enumerate(self.encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
        
        # 一次编码 [B,A,Bp]，另一次编码 AB，但两次使用相同权重。
        # 其中 z_e_BABp[:B_IN] 即 E(B)，下一阶段直接复用它；没有第二套专门给 MAE 的 B encoder。
        z_e_BABp = get_encoded(z_BABp, is_selected_BABp, is_mask_encoder_BABp)
        z_e_AB = get_encoded(z_AB, is_selected_AB, is_mask_encoder_AB)

        # ④ 只将 B 和 AB 送入 MAE decoder；A/Bp 在这里不走 D，而在后面走表示头。
        # 解码 batch 顺序固定为 [B, AB_1..AB_R]，后面 PMI 比较据此取首组与其余组。
        # Decoder
        is_mask_BAB = torch.cat([is_mask_B, is_mask_AB], dim=0)
        # 取 z_e_BABp 的首个 B_IN 切片，直接复用上一步的 B 编码并放回原空间位置。
        z_d_B_nomaskpos = z_e_BABp.new_zeros((B_IN,N,d)).masked_scatter(
            repeat(is_selected_B, 'B N -> B N d', d=d),
            z_e_BABp[:B_IN, self.in_context_len:].flatten(),
        )
        # AB 同样先把 selected 的编码还原到 N 个空间位置，未选位置先留零。
        z_d_AB_nomaskpos = z_e_AB.new_zeros((R*B_IN,N,d)).masked_scatter(
            repeat(is_selected_AB, 'B N -> B N d', d=d),
            z_e_AB[:, self.in_context_len:].flatten(),
        )
        z_d_BAB_nomaskpos = torch.cat([z_d_B_nomaskpos, z_d_AB_nomaskpos], dim=0)
        # in-context 编码保留；所有逻辑 masked 位置统一换成 mask_token，包含 encoder 中的隔离补齐项。
        z_d = torch.cat([
            torch.cat([z_e_BABp[:B_IN, :self.in_context_len], z_e_AB[:, :self.in_context_len]], dim=0),
            torch.where(
                repeat(is_mask_BAB, 'RB N -> RB N 1'),
                repeat(self.mask_token.to(dtype), '1 1 d -> (R B) N d', R=1+R, B=B_IN, N=N),
                z_d_BAB_nomaskpos
            )
        ], dim=1)
        # B 与 AB 各自使用自己的可见中心相对位置，再执行完整网格 self-attention 得到每个位置的条件。
        z_BAB_factor_pos = torch.cat([z_BABp_factor_pos[:B_IN], z_AB_factor_pos], dim=0)
        z_d = z_d + torch.cat([z_BAB_factor_pos.new_zeros((1+R)*B_IN, self.in_context_len, d), z_BAB_factor_pos], dim=1)
        # 此 decoder 使用完整原网格 RoPE，CLS 默认复用首个网格位置；未显式传入可见中心 cls_idx。
        for block_idx, block in enumerate(self.decoder):
            z_d = block(z_d, self.feat_rope_incontext, attn_mask=None)

        # ⑤ 联合步骤的 MAR 同时训练 B 和 AB 两种条件，不是只训练 AB。
        # B 组在 mask_B 位置训练；AB 组先取 mask_A，再用 mask_B 筛选，最终只剩 mask_A∩mask_B。
        # MAR loss
        # Tradeoff: This wastes compute by calculating B tokens during A mask. But necessary for constant-size compilable masks
        # Masking is coupled here, but correct because ultimately is_mask_B valid check and is_mask_A masking combine to work
        is_mask_BA = torch.cat([is_mask_B, is_mask_A], dim=0)
        # 去掉 in-context 前缀后选目标位置条件，再沿 O 复制；Y 是当前全部候选目标位置的展平计数。
        z_m = repeat(z_d[:,self.in_context_len:][is_mask_BA], 'Y d -> (O Y) d', O=O)
        # 按相同 [B,AB_1..AB_R] 顺序复制干净目标，保证目标索引和 decoder 条件逐项对齐。
        feats_p_rb = repeat(x_p_target, 'B N d -> (R B) N d', R=R1)
        feats_p_m = repeat(feats_p_rb[is_mask_BA], 'Y d -> (O Y) d', O=O, d=feats_p_rb.size(-1))
        # MAR 用 10% 均匀 + 90% logit-normal 时间；与下面 PMI 估计仅用 logit-normal 的采样略有不同。
        t_m = torch.where(torch.rand(z_m.size(0), 1, device=device) < 0.1, torch.rand(z_m.size(0), 1, device=device), torch.sigmoid(torch.randn(z_m.size(0), 1, device=device) * self.std_mar + self.mu_mar))
        eps_m = torch.randn_like(feats_p_m)
        # 真实代码时间方向：t=0 噪声，t=1 干净；MLP 输出回归 clean target，不回归 epsilon。
        feats_p_m_t = t_m * feats_p_m + (1-t_m) * eps_m
        feats_pred = self.diff_mlp(feats_p_m_t, t_m.flatten(), z_m)
        # 额外计算但对 AB 已可见的目标以 mask 归零；两边同时乘 mask 后再算 Huber，确保无损失贡献。
        mse_is_valid = repeat(repeat(is_mask_B, 'B N -> (R B) N', R=R1)[is_mask_BA], 'Y -> (O Y) 1', O=O)
        # 2*Huber(delta=10) 在小残差区等于 MSE，较大残差梯度受限；这里还对时间权重与有效比例归一化。
        mse_loss = 2*(F.huber_loss(feats_pred*mse_is_valid, feats_p_m*mse_is_valid, reduction='none', delta=10) / (1-t_m).clamp_min(0.33).pow(2)).mean() / mse_is_valid.float().mean()

        # ⑥ 用同一生成器比较 B 与 AB 对目标 p 的预测误差，构造表示学习的无梯度监督。
        # no_grad 只覆盖这一段 PMI target 计算，不代表 E/D/MLP 被冻结；上面的 MAR 仍训练它们。
        # Likelihood Integration
        with torch.no_grad():
            # 先恢复 [R1,B_IN,N,d] 的条件维，再在每图的 P 个目标坐标取出 [R1,P,B_IN,d]。
            z_decoder_r = rearrange(z_d[:,self.in_context_len:], '(R B) N d -> R B N d', R=R1)
            z_bp_rp = z_decoder_r[:, torch.arange(B_IN, device=device), is_patch_target.int().argmax(-1)].float()
            # 沿 V 复制条件并展平为 [V*P*R1*B_IN,d]；同一次 MLP 调用处理所有条件，之后再恢复维度。
            z_bp_vpr = repeat(z_bp_rp, 'R P B d -> (V P R B) d', V=V)

            # 干净目标也按 [V,P,R1,B_IN] 对齐；同一 p 在不同条件下的目标值相同。
            x_bp_vpr = repeat(repeat(x_p_target, 'B N d -> P B N d', P=P)[is_patch_target], '(P B) d -> (V P R B) d', P=P, R=R1, V=V).float()
            # 时间先采样为 [V,P,B_IN]，再沿 R1 复制；噪声也先采样再沿 R1 复制。
            # 因此 B 与每个 AB 比较时使用同一个 p_t、同一个 t 和同一个 epsilon，只更换 MAE 条件。
            t_bp_vp = torch.sigmoid(torch.randn(V,P,B_IN, device=device) * self.std_repr + self.mu_repr)
            t_bp_vpr = repeat(t_bp_vp, 'V P B -> (V P R B) 1', R=R1).float()
            eps_bp_vpr = repeat(torch.randn(V,P,B_IN,x_bp_vpr.size(-1),device=device), 'V P B d -> (V P R B) d', R=R1).float()
            # 带噪目标直接进入 MLP，不重新送回 MAE；MAE 条件已由各自的可见上下文算好。
            x_bp_vpr_noisy = t_bp_vpr * x_bp_vpr + (1-t_bp_vpr) * eps_bp_vpr

            # MLP 用当前训练精度前向，预测与误差统计转为 float32；随后 h 又转回 dtype，保留作者的数值路径。
            x_bp_vpr_pred = self.diff_mlp(x_bp_vpr_noisy.to(dtype), t_bp_vpr.flatten().to(dtype), z_bp_vpr.to(dtype)).float()

            # 实现权重为 0.5*t/max(1-t,0.33)^2，通道误差用平方和而不是通道均值。
            # 这是带时间采样、截断和有限 V 的 PMI 估计量；不应把它描述为精确似然或一次普通 MSE。
            freq_weight = 1/2*t_bp_vpr/(1-t_bp_vpr).clamp_min(0.33).pow(2)
            h_p_vprb = freq_weight * (x_bp_vpr_pred - x_bp_vpr).pow(2).sum(-1, True)
            # 整理为 [R1,B_IN,P,V]，保留最后的 V 维，直到 B/AB 相减时才做 Monte Carlo 平均。
            h_p_rbpv = rearrange(h_p_vprb, '(V P R B) 1 -> R B P V', P=P, R=R1, V=V).to(dtype)

        # ⑦ 表示路线只接收 E(B/A/Bp)，不接收 MAE decoder 的条件或带噪目标。
        # REPR Calculation
        # 先追加表示头的 3 层 self-attention，继续使用同样的补齐隔离规则。
        def get_repr_encoder(z_e, is_selected, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic)
            for block_idx, block in enumerate(self.repr_encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
            
        # 再以扩展网格的位置 embedding 作为 query，通过 3 层 cross-attention 读取可见上下文。
        def get_repr_decoder(z_e, repr_factor_pos_pad, factorpad_rope_idx, encoder_rope_idx, cls_rope_idx, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)

            # K/V 侧只开放真正可见 patch 与 in-context tokens；隔离补齐项即使存在于张量，也不会被查询读取。
            attn_mask = rearrange(~is_mask_encoder_ic, 'B K -> B 1 1 K')
            
            # 初始空间 query 是位置 embedding，未拼入图像真值；内容通过 cross-attention 从上下文获得。
            repr_tokens = repr_factor_pos_pad
            # Q 用扩展网格坐标；K 的 patch 保持对应网格位置，而 in-context K 的 RoPE 在此显式使用可见中心。
            q_rope = partial(self.feat_rope_incontext_pad, idx=factorpad_rope_idx, cls_idx=None, use_cls=False)
            k_rope = partial(self.feat_rope_incontext_pad, idx=encoder_rope_idx, cls_idx=cls_rope_idx, use_cls=True)
            for block in self.representation_encoder:
                repr_tokens = block(repr_tokens, z_e, q_rope, k_rope, attn_mask=attn_mask)
            return repr_tokens

        # 所有 B/A/Bp 视图经过同一个表示头；AB 只为生成条件服务，不参与此处表示点积。
        z_re_BABp = get_repr_encoder(z_e_BABp, is_selected_BABp, is_mask_encoder_BABp)

        z_BABp_repr = get_repr_decoder(z_re_BABp, z_BABp_factor_pos_pad, z_BABp_factorpad_rope_idx, z_BABp_encoder_rope_idx[is_selected_BABp].view((1+R+P)*B_IN,-1), z_BABp_cls_rope_idx, is_mask_encoder_BABp)
        # Q=self.pad_hw²，D_repr=self.repr_dim；空间表示排列为 [R1,B_IN,1,Q,D_repr] 与 [B_IN,P,Q,D_repr]。
        # 它们保留整张查询网格，后面点积会跨 Q 和 D_repr 求和，并非只拿 p 所在位置做相似度。
        z_BA_repr = rearrange(self.repr_final_layer(z_BABp_repr[:(1+R)*B_IN]), '(R b) q d -> R b 1 q d', R=R1)
        z_Bp_repr = rearrange(self.repr_final_layer(z_BABp_repr[(1+R)*B_IN:]), '(P b) q d -> b P q d', P=P)
        # 全局表示来自 repr_encoder 后的 in-context tokens，经独立投影再将 [context_len,repr_dim] 展平。
        z_repr_global = rearrange(self.repr_final_layer_global(z_re_BABp[:(1+R)*B_IN, :self.in_context_len]), '(R b) C d -> R b 1 (C d)', R=R1)
        z_Bp_repr_global = rearrange(self.repr_final_layer_global(z_re_BABp[(1+R)*B_IN:, :self.in_context_len]), '(P b) C d -> b P (C d)', P=P)

        # pad_dropout 只把 B/A 空间表示的外圈查询置零，不是随机丢弃输入 patch，也不做常规 dropout 的反向缩放。
        # 随机决策按 R1 组广播到整组图像和查询；Bp 与全局表示在这一行不作同样处理。
        z_BA_repr = torch.where(torch.rand(z_BA_repr.size(0),device=device).view(-1,1,1,1,1) < self.pad_dropout, z_BA_repr*self.pad_dropout_mask.to(dtype), z_BA_repr)

        # 空可见集合的 B/A 表示显式置零，防止仅凭可学习 query/in-context 常量产生非零空集合贡献。
        is_empty_BA = is_mask_BA.all(-1)
        z_BA_repr = torch.where(is_empty_BA.view(R1, B_IN, 1, 1, 1), 0, z_BA_repr)
        z_repr_global = torch.where(is_empty_BA.view(R1, B_IN, 1, 1), 0, z_repr_global)
        # 拆回 B 与 R 个 A；同一个 B 表示会与 P 个 Bp、R 个 A 配对，不需要重复编码 B。
        z_B_repr = z_BA_repr[:1]
        z_A_repr = z_BA_repr[1:]
        z_B_repr_global = z_repr_global[:1]
        z_A_repr_global = z_repr_global[1:]

        # ⑧ 用表示差的点积回归第⑥步的 PMI 目标，并分别训练空间表示和全局表示。
        # PMI Loss
        # 有效性要求 p 对 B 和当前 A 都被遮住，保证比较的是“加入 A 对未知 p 的帮助”。
        # 即使采样时尽量选共同遮挡位置，这个检查也不能省略；无效配对最终不计入损失。
        is_valid_pmi = reduce(~(repeat(~is_valid_target, '(R b) n -> R b P n', R=R1, P=P) & repeat(is_patch_target, 'P b n -> R b P n', R=R1)), 'R b P n -> R b P', 'all')  # don't allow loss for: ~is_mask and is_bp
        is_valid_B = is_valid_pmi[:1]
        is_valid_A = is_valid_pmi[1:]
        is_valid = is_valid_B & is_valid_A

        # 第 0 条条件是 B，后 R 条是 AB；B 的误差在广播后供每一个 A 使用。
        h_p_given_B_preintegration = h_p_rbpv[:1]
        h_p_given_BA_preintegration = h_p_rbpv[1:]

        # 空间预测是 <R(A), R(Bp)-R(B)>，对全部查询与通道求和，并保留源码中的宽度/查询数缩放。
        # 这里 d 是 backbone 宽度，未必等于 repr_dim；原作者 TODO 保留，不把两者擅自替换。
        # 全局表示也做同样的增量点积，但有独立的输出头与归一化因子。
        pmi_pred = torch.einsum('rbqd,bpqd->rbp', z_A_repr[:,:,0], z_Bp_repr - z_B_repr[0]) / (2 * z_A_repr.size(-1) * z_A_repr.size(-2)**0.5) * d**0.5  # GIGA TODO USE REPR HERE!
        pmi_pred_global = torch.einsum('rbd,bpd->rbp', z_A_repr_global[:,:,0], z_Bp_repr_global - z_B_repr_global[0]) / (2 * z_A_repr_global.size(-1)**0.5)
        # 方向固定为 h(p|B)-h(p|AB)，再沿 V 平均；加入 A 改善预测时趋向正，也允许负值，不做截零。
        pmi_target = (h_p_given_B_preintegration - h_p_given_BA_preintegration).mean(-1)

        # 实际表示损失是 2*Huber(delta=repr_delta)，在小误差区等于平方误差，超出阈值则限制梯度。
        # 先按每个 [A重复,图像] 的有效 p 比例校正，再整体求均值；clamp 防止整组无有效 p 时除零。
        # Using Huber for MSE Gradient Clipping
        repr_loss = (2*F.huber_loss(pmi_pred, pmi_target, reduction='none', delta=self.repr_delta) * is_valid / reduce(is_valid.bfloat16(), 'R b P -> R b 1', 'mean').clamp_min(1e-5)).mean()
        repr_loss_global = (2*F.huber_loss(pmi_pred_global, pmi_target, reduction='none', delta=self.repr_delta) * is_valid / reduce(is_valid.bfloat16(), 'R b P -> R b 1', 'mean').clamp_min(1e-5)).mean()

        # 总损失含 MAR + λ×空间 RePR + λ×全局 RePR；PMI target 已无梯度，表示损失只沿预测侧回传。
        loss = mse_loss + self.repr_loss_weight * repr_loss + self.repr_loss_weight * repr_loss_global
        # 返回的首项保留计算图供 backward；后面三项 detach+clone 仅作日志指标，不参与额外梯度。
        return loss, mse_loss.detach().clone(), repr_loss.detach().clone(), repr_loss_global.detach().clone()


    # 特征导出路线：像素 → 共享 encoder → 表示头 → 空间/全局特征，不运行 MAE decoder 或 diffusion MLP。
    # 默认 num_mask_tokens=0 看完整图；本方法未加 no_grad，是否求梯度由调用方决定。
    @torch.compile
    def forward_feats(self, x, num_mask_tokens=0, return_padded=True):
        assert self.in_context_len > 0

        B_IN, C, H, W = x.shape
        device = x.device
        x_p_og = self.patchify(x)
        z_og_nopos = self.x_embedder(x_p_og)
        dtype = z_og_nopos.dtype

        N = self.num_patches
        d = self.hidden_size
        R=1

        # 可选随机遮挡用于提取部分可见图的表示；这里 selected 恰为可见集合，不使用训练时的定长补齐。
        mask_rank = torch.rand((B_IN, N), device=device).argsort(dim=1).argsort(dim=1)
        is_mask_rb = mask_rank < num_mask_tokens
        is_selected_rb = ~is_mask_rb
        is_mask_encoder_rb = rearrange(is_mask_rb[is_selected_rb], '(B m) -> B m', B=B_IN)

        # 位置构造与联合训练对应：加性 embedding 相对可见中心，RoPE 则使用显式的扩展网格坐标。
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

            # return_padded=True 保留外圈查询，16×16 输入返回 324 个空间位置。
            # False 只在原始 256 个 patch 位置建立查询；用于下一轮教师目标时要与原图 patch 一一对应。
            if return_padded:
                return factor_pos, factor_pos_pad, patch_rope_idx, pad_rope_idx, cls_rope_idx
            else:
                return factor_pos, factor_pos, patch_rope_idx, patch_rope_idx, cls_rope_idx
        
        z_factor_pos, z_factor_pos_pad, z_patch_rope_idx, z_pad_rope_idx, z_cls_rope_idx = get_factor_pos(is_mask_rb)

        z = repeat(z_og_nopos, 'B N d -> (R B) N d', R=R) + z_factor_pos
                
        # 复用同一 E 的编码逻辑；in-context tokens 仍在序列前缀，训练好的参数不会在此另起一套。
        def get_encoded(z, is_selected, is_mask_encoder):
            B_ = z.size(0)
            z_e = z[is_selected].view(B_, -1, d)  # e: encoder
            z_e = torch.cat([repeat(self.in_context_tokens.to(dtype), '1 l d -> b l d', b=B_), z_e], dim=1)
            is_mask_encoder_ic = torch.cat([is_mask_encoder.new_zeros(B_, self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic)
            for block_idx, block in enumerate(self.encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e
        
        z_e = get_encoded(z, is_selected_rb, is_mask_encoder_rb)

        # 表示头的 3 层 self-attention；其 CLS RoPE 沿用 selected 中首个可见位置的规则。
        def get_repr_encoder(z_e, is_selected, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)
            attn_mask_encoder = self.get_attention_mask(is_mask_encoder_ic)
            for block_idx, block in enumerate(self.repr_encoder):
                z_e = block(z_e, partial(self.feat_rope_incontext, m=~is_selected, m_selected=is_mask_encoder), attn_mask=attn_mask_encoder)
            return z_e

        z_re = get_repr_encoder(z_e, is_selected_rb, is_mask_encoder_rb)

        # 空间查询 cross-attention 的 K/V 来自表示头前半段；与训练一样屏蔽被遮挡的上下文项。
        def get_repr(z_e, repr_factor_pos_pad, factorpad_rope_idx, encoder_rope_idx, cls_rope_idx, is_mask_encoder):
            is_mask_encoder_ic = torch.concatenate([is_mask_encoder.new_zeros(is_mask_encoder.size(0), self.in_context_len), is_mask_encoder], dim=1)
            
            # 下列 same_rope 分支准备了“只读同坐标 patch”的限制掩码，但是否启用取决于 q_is_restricted。
            same_rope = rearrange(factorpad_rope_idx, 'B Q -> B Q 1') == rearrange(encoder_rope_idx, 'B K -> B 1 K')
            same_rope_or_in_context = F.pad(same_rope, (self.in_context_len, 0), value=False)
            q_has_k = reduce(same_rope, 'B Q K -> B Q 1', 'any')
            # 当前阈值写死为 0.0，所以对 torch.rand_like 的非负样本此条件恒为 False。
            # 因此实际使用 attn_mask_default，保留原作者这段代码及其随机数调用，不在注释任务中删除或修复。
            q_is_restricted = q_has_k & (torch.rand_like(q_has_k, dtype=dtype) < 0.0)
            attn_mask_default = rearrange(~is_mask_encoder_ic, 'B K -> B 1 K')
            attn_mask_restricted = rearrange(~is_mask_encoder_ic, 'B K -> B 1 K') & ((~q_is_restricted) | same_rope_or_in_context)
            attn_mask = repeat(torch.where(q_is_restricted, attn_mask_restricted, attn_mask_default), 'B Q K -> B H Q K', H=self.num_heads)
            
            # 初始 query 为位置 embedding；此处 K 的 in-context RoPE 显式放在可见中心，区别于前面的 self-attention。
            repr_tokens = repr_factor_pos_pad
            q_rope = partial(self.feat_rope_incontext_pad, idx=factorpad_rope_idx, cls_idx=None, use_cls=False)
            k_rope = partial(self.feat_rope_incontext_pad, idx=encoder_rope_idx, cls_idx=cls_rope_idx, use_cls=True)
            for block in self.representation_encoder:
                repr_tokens = block(repr_tokens, z_e, q_rope, k_rope, attn_mask=attn_mask)
            return repr_tokens

        # 空间结果形状 [B_IN,Q,repr_dim]；全局结果将各 in-context 输出展平为 [B_IN,context_len*repr_dim]。
        z_repr = self.repr_final_layer(get_repr(z_re, z_factor_pos_pad, z_pad_rope_idx, z_patch_rope_idx[is_selected_rb].view(R*B_IN,-1), z_cls_rope_idx, is_mask_encoder_rb))
        z_repr_global = rearrange(self.repr_final_layer_global(z_re[:, :self.in_context_len]), 'RB C d -> RB (C d)')

        return z_repr, z_repr_global


# 各型号保持 18 层共享 encoder 与 6 层 MAE decoder，主要通过宽度和 head 数缩放。
# 表示头另含 3 层 self-attention + 3 层 cross-attention；导出特征时走 E+表示头，不走这 6 层 MAE decoder。
# patch_size 均为 16；diff_mlp_depth、repr_dim、训练重复数等仍可由 kwargs/训练参数覆盖。
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

# 模型名称到工厂函数的映射，供 denoiser_repr.py 按命令行 --model 实例化。
REPR_MODELS = {
    'REPR-XT/16': REPR_XT_16,
    'REPR-T/16': REPR_T_16,
    'REPR-S/16': REPR_S_16,
    'REPR-B/16': REPR_B_16,
    'REPR-L/16': REPR_L_16,
}
