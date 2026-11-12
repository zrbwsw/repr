**Figure 1 里的 **$ B $** 和 **$ AB $**，都进入 MAE。这里的 MAE 是生成预测分支中的 encoder＋decoder；它先输出目标位置的条件特征，再由 diffusion MLP 预测目标 patch。**

**这个 MAE 的 encoder，与上方计算 **$ R(A) $**、**$ R(B) $**、**$ R(Bp) $** 的 encoder 是共享的**。代码直接复用了已经算好的 $ E(B) $。

我核对了仓库的 [cb1c132](https://github.com/zrbwsw/repr/tree/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3) 版本，重点看了 `model_repr.py` 的 `forward_repr()` 和 `denoiser_repr.py` 的训练调度。下面先按第一轮“预测像素 patch”的实现展开。

## 1. 先把 Figure 1 中的三个模块分清楚
$ \boxed{\text{图中的 MAE}=E_\theta+D_\theta} $

$ \boxed{\text{完整的生成预测分支}=E_\theta+D_\theta+M_\theta} $

$ \boxed{\text{表征分支 }R_\theta=E_\theta+H_\theta} $

这里用 $ H_\theta $ 统称代码中的表征头。**两条分支共用同一个 **$ E_\theta $**，之后分别进入生成 decoder 或表征头。**

在真实代码中，对应关系如下：

| Figure 1 中的模块 | 代码对应 | 输出或作用 |
| --- | --- | --- |
| Encoder $ E_\theta $ | `self.encoder` | 编码当前视图的可见上下文 |
| MAE Decoder $ D_\theta $ | `self.decoder` | 为各个位置生成条件特征 |
| Diffusion MLP $ M_\theta $ | `self.diff_mlp` | 根据条件特征、带噪 patch 和时间，预测干净 patch |
| 表征头 $ H_\theta $ 的前半部分 | `self.repr_encoder` | 额外 3 层 self-attention |
| 表征头 $ H_\theta $ 的后半部分 | `self.representation_encoder` | 3 层 cross-attention，产生空间表征 |


这些模块都在同一个 `REPR` 类中定义。[源码：模块定义](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L542-L592)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/58315253/1794469225312-b1e4db5d-8559-417d-bac3-fb2af7074d2c.png)

```plain
E -->|"A、B、Bp 的编码"| H["表征头 H"]
E -->|"B、AB 的编码"| D["MAE Decoder D"]

H --> R["R(A)、R(B)、R(Bp)"]
R --> S["表征分数：A 与 Bp−B 的点积"]

D --> Z["两种条件特征：z(p|B)、z(p|AB)"]
Z --> M["Diffusion MLP M"]
N["相同的带噪 patch 和时间"] --> M
M --> T["加权预测误差：B 条件减 AB 条件"]

S --> L["RePR loss"]
T -->|"stop-gradient，作为目标"| L -->
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/__mermaid_v3/b61dcb3bce87ec0d18e50a05bd00c6e4.svg)

**因此，Figure 1 下方的 MAE 负责产生“用于预测的条件”，上方的 **$ R $** 负责产生“用于计算关系的表征”。它们在 encoder 处连接起来。**[源码：共享编码、生成条件与表征计算](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L995-L1120)

## 2. $ B $、$ AB $、$ Bp $ 分别是什么，为什么进入不同分支？
先用一个具体的 patch 编号例子。假设：

$ A=\{4,7\},\qquad B=\{1,4\},\qquad p=9. $

那么：

$ AB=A\cup B=\{1,4,7\}, $

$ Bp=B\cup\{p\}=\{1,4,9\}. $

这里的 $ AB $ 表示**可见 patch 的并集**。$ A $ 与 $ B $ 可以重叠，例如这里都包含 patch 4。

各个视图在联合训练步中的用途是：

| 视图 | 例子中的可见 patch | 进入共享 $ E $ | 进入生成 decoder $ D $ | 进入表征头 $ H $ |
| --- | --- | ---: | ---: | ---: |
| $ A $ | $ \{4,7\} $ | ✓ | — | ✓，得到 $ R(A) $ |
| $ B $ | $ \{1,4\} $ | ✓ | ✓，得到预测 $ p $ 的条件 | ✓，得到 $ R(B) $ |
| $ AB $ | $ \{1,4,7\} $ | ✓ | ✓，得到预测 $ p $ 的条件 | — |
| $ Bp $ | $ \{1,4,9\} $ | ✓ | — | ✓，得到 $ R(Bp) $ |


代码中的 mask 使用 `True` 表示遮挡，所以可见集合的并集写成：

```python
is_mask_AB = is_mask_A & repeat(is_mask_B, 'B N -> (R B) N', R=R)
```

它的意思是：**只有同时被 **$ A $** 和 **$ B $** 遮住的位置，才继续在 **$ AB $** 中被遮住。**[源码：mask 构造](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L780-L814)

### 为什么生成分支要比较 $ B $ 和 $ AB $？
因为它要回答：

> **已经知道 **$ B $** 的情况下，再给模型看 **$ A $**，会让同一个目标 patch **$ p $** 更容易预测多少？**
>

对应的理想概率比是：

$ \log P(p\mid A,B)-\log P(p\mid B). $

所以需要两种上下文：

+ 只给 $ B $，预测 $ p $；
+ 给 $ A\cup B $，再预测同一个 $ p $。

这就是图中 $ B $ 和 $ AB $ 进入 MAE 的原因。

而上方的表征分支，要把这种“额外预测信息”表达成向量关系：

$ \left\langle R(A),\,R(Bp)-R(B)\right\rangle. $

因此，上方需要 $ A $、$ B $、$ Bp $ 三个视图，下方需要 $ B $、$ AB $ 两种生成条件。真实代码最后也是把这两边构造出的分数进行匹配。[源码：表征分数与监督目标](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1103-L1120)

这里还要补一个实现细节：代码最终通过 `is_valid` 筛选，**只有满足 **$ p\notin A\cup B $** 的有效配对才计入 PMI loss**；不能笼统地说采样阶段的每一个 $ A $ 都已经排除了所有候选 $ p $。

## 3. 最直接的代码证据：$ E(B) $ 如何被两条分支复用？
`forward_repr()` 首先把 $ B $、$ A $、$ Bp $ 的 mask 沿 batch 维拼起来：

```python
is_selected_BABp = torch.cat(
    [is_selected_B, is_selected_A, is_selected_Bp], 0
)

is_mask_BABp = torch.cat(
    [is_mask_B, is_mask_A, is_mask_Bp], 0
)
```

随后执行：

```python
z_e_BABp = get_encoded(
    z_BABp, is_selected_BABp, is_mask_encoder_BABp
)

z_e_AB = get_encoded(
    z_AB, is_selected_AB, is_mask_encoder_AB
)
```

两个 `get_encoded()` 内部都遍历同一个 `self.encoder`。[源码：视图打包与共享 encoder](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L944-L1007)

### 注意 `BABp` 这个变量名
`z_e_BABp` 的意思是：

> 一个 batch 中依次装着 $ B $、$ A $、$ Bp $ 三类视图的编码结果。
>

它不表示把这些视图合成一张图。每个 batch 样本内部独立做 attention。

接下来，代码从这批结果中取出：

```python
z_e_BABp[:B_IN]
```

这部分就是 $ E(B) $。其中 `B_IN` 是输入 batch size。

同一份 $ E(B) $ 被用于：

1. **生成分支**：取出后补回空间位置，送入 `self.decoder`。
2. **表征分支**：作为整个 `z_e_BABp` 的一部分，继续进入表征头。

而 $ E(AB) $ 也会送入同一个 `self.decoder`。也就是说，$ E(B) $ 和 $ E(AB) $ 都由 MAE decoder 处理。[源码：decoder 使用 ](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1009-L1031)$ E(B) $[ 与 ](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1009-L1031)$ E(AB) $

所以，对你的疑问最直接的回答是：

> $ B $** 的确进入 MAE；同时，$ B $ 的 encoder 结果也用于计算 $ R(B) $。Figure 1 把这两种用途画在了不同位置，源码则明确复用了共享计算。**
>

## 4. $ B $ 和 $ AB $ 进入 MAE 后，究竟输出什么？
以 $ B $ 分支为例，逻辑上分三步。

### 第一步：Encoder 编码可见上下文
$ h_B=E_\theta(B). $

在前面的例子中，它能使用的内容来自 patch 1、4。

### 第二步：Decoder 补入 mask token，恢复各个空间位置
代码先将 encoder 输出放回对应的 patch 位置；被遮住的位置使用 `self.mask_token`，随后加入位置编码，运行 decoder。

核心操作是：

```python
torch.where(
    repeat(is_mask_BAB, 'RB N -> RB N 1'),
    repeat(self.mask_token.to(dtype), '1 1 d -> (R B) N d',
           R=1+R, B=B_IN, N=N),
    z_d_BAB_nomaskpos
)
```

因此，目标位置 $ p $ 虽然没有真实内容，也有一个可以从上下文读取信息的 decoder token。[源码：补 mask token 与 decoder 前向](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1020-L1031)

### 第三步：取出目标位置的条件特征
$ z_{p\mid B}=\big[D_\theta(E_\theta(B))\big]_p, $

$ z_{p\mid AB}=\big[D_\theta(E_\theta(A\cup B))\big]_p. $

这里把补 mask、位置编码等操作包含在 $ D_\theta $ 的记号里。

**这两个 **$ z $** 都是 hidden feature。它们表达的是：“根据当前可见上下文，目标位置 **$ p $** 应该如何预测。”**

随后，代码才从 decoder 输出中选取目标位置，并传给 diffusion MLP。[源码：选择目标位置的 decoder 特征](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1047-L1059)

[源码：固定长度选择](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L800-L811)、[attention 分组](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L667-L670)、[表征头可见性限制](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1073-L1082)。

### 补充：这里的 MAE decoder 是预训练好的吗？
不是外部加载的预训练 MAE decoder。第一轮启动时，`self.decoder` 和 `self.diff_mlp` 都在当前 `REPR` 模型中创建，并通过 `initialize_weights()` 初始化；随后它们和共享 encoder 一起由当前训练目标更新。也就是说，图中的 MAE decoder 是**联合训练的生成预测头**，不是先单独预训练再冻结的模块。

代码里确实有一个 `frozen_net`，但它只在第二轮 `repr_mar_featspred` 中作为上一轮 checkpoint 的 teacher 使用；这和第一轮的 decoder 是否预训练是两件事。第二轮的 teacher 只用 `forward_feats()` 产生空间表征目标，具体交接放在第 9 节。

## 5. 带噪的 $ p $ 在哪里进入？它单独进入 diffusion MLP
### 5.1 目标 $ p $ 不是每次前向只选一个
这里的“目标 patch $ p $”是**一张图像中的一个 patch 坐标**，不是一个包含整张图的张量。一次 `forward_repr()` 会按配置为每张图选 $ P=\texttt{p\_repeat} $个候选坐标，并行计算它们；代码默认值与实验命令可以不同，README 第一轮命令常用 $ P=16 $。因此更准确的说法是：**每个候选分支有一个 **$ p $**，但一次 batch 同时处理多个 **$ p $**。**

还要区分两类“目标位置”：MAR 项会对 $ B $/ $ AB $ 条件下逻辑上被 mask 的各个位置进行去噪训练，并不只训练这 $ P $ 个候选；$ P $ 个 $ p $ 主要是 PMI/RePR 对比专用的增量位置。PMI 段才通过 `is_patch_target.argmax(-1)` 取出每个候选 $ p $ 的 decoder 条件，并在 $ V=\texttt{diff\_repeat\_repr} $ 次时间/噪声样本上比较 $ B $ 与 $ AB $。

#### $ p $ 是怎样选出来的？
先随机生成 $ A $、$ B $ 的 mask。`sample_mask_random` 会优先从“在相关 $ A $ 和 $ B $ 中都不可见”的位置中按随机顺序挑选候选；`sample_mask_block` 则在块状 mask 下按跨多个 $ A $ 视图的可见次数排序，优先选择没有被 $ A $ 看见的位置。候选不足时，代码会补充位置，但最后的 `is_valid`/`is_valid_pmi` 仍会过滤掉不满足 PMI 条件的配对。

代码中的 `is_bp` 是形状为 $ [P,B,N] $ 的 one-hot 目标掩码，每个 $ P $ 分支只置一个 $ p $ 坐标。随后构造：

$ B_p=B\cup\{p\},
\qquad
\texttt{is\_mask\_Bp}=\texttt{is\_mask\_B}\;\&\;(\neg\texttt{is\_bp}). $

因此，$ B_p $ 不是“把 $ p $ 替换成一张新图”，而是让 $ B $ 多看一个 patch。$ B_p $ 中新增的坐标使用目标 patch 的 embedding，并加上 `is_b_embed` 这个角色标记；其它位置仍使用原来的图像 patch embedding。

#### 第一轮的维度：目标是像素 patch 向量
设输入为 $ 256\times256 $ 的 RGB 图像，patch 大小为 $ 16\times16 $，则：

$ N=\left(\frac{256}{16}\right)^2=256,
\qquad
\texttt{patch\_dim}=3\times16\times16=768. $

`patchify(x)` 的输出为：

$ x_p^{\mathrm{og}}\in\mathbb{R}^{B\times N\times768}. $

第一轮 `x_target is None`，所以：

$ x_{p,\mathrm{target}}=x_p^{\mathrm{og}},
 \qquad
 z_{\mathrm{target}}=\texttt{x\_embedder}(x_p^{\mathrm{og}})\in\mathbb{R}^{B\times N\times d}, $

其中 $ d $ 是 encoder hidden size。选出 $ P $ 个目标坐标后，干净目标的形状可写成 $ [P,B,768] $；再复制到 $ V $ 次 Monte Carlo 时间/噪声样本和 $ R_1=R+1 $ 个条件分支后，代码会展平为：

$ [V\,P\,R_1\,B,768]. $

因此第一轮 diffusion MLP 的输入和输出都是 768 维的 RGB patch 向量。它在数学上是一个向量去噪问题，但不能把第一轮的目标严格称为“encoder latent”；真正的 latent hidden 条件是 $ z_{p\mid B} $ 或 $ z_{p\mid AB} $。

#### 第二轮才切换到 latent feature target
第二轮传入冻结 teacher 产生的 `x_target`，其形状为 $ [B,N,\texttt{repr\_dim}] $。此时：

$ x_{p,\mathrm{target}}=x_{\mathrm{target}},
 \qquad
 z_{\mathrm{target}}=\texttt{feats\_embedder}(x_{\mathrm{target}}), $

并且 `diff_mlp` 的输入/输出从 `patch_dim` 改成 `repr_dim`。所以：

+ 第一轮：$ M_\theta $ 预测 RGB patch，目标维度为 `patch_dim`；
+ 第二轮：学生模型仍用相同的去噪结构，但预测 teacher 的空间表征，目标维度为 `repr_dim`；
+ 两轮中 $ A $、$ B $ 的上下文仍由当前输入图像的像素 patch embedding 构造，改变的只是 $ p $ 的 target space 和 $ B_p $ 中新增目标的 embedding。

同一个候选 $ p $ 的 $ B $ 与 $ AB $ 条件共享相同的 $ t $ 和噪声 $ \epsilon $，从而只比较上下文条件的差异；这不意味着整批样本只有一个 $ p $。

[源码：候选目标构造与 ](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L780-L814)$ B_p $[ mask](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L780-L814)

现在有了两种条件特征，再对**同一个目标 patch** 加噪。

实际代码使用的时间约定是：

$ p_t=t\,p+(1-t)\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I). $

+ $ t=0 $：纯噪声；
+ $ t=1 $：干净 patch。

```python
x_bp_vpr_noisy = (
    t_bp_vpr * x_bp_vpr
    + (1-t_bp_vpr) * eps_bp_vpr
)
```

接下来分别计算：

$ \hat p_B=M_\theta(p_t,t,z_{p\mid B}), $

$ \hat p_{AB}=M_\theta(p_t,t,z_{p\mid AB}). $

[源码：加噪与 diffusion MLP 调用](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1053-L1062)

### MLP 的三个输入在源码中很清楚
`SimpleMLPAdaLN.forward(self, x, t, c)` 的主体是：

```python
x = self.input_proj(x)
t = self.time_embed(t)
c = self.cond_embed(c)

y = t + c

for block in self.res_blocks:
    x = block(x, y)

return self.final_layer(x, y)
```

分别对应：

| 参数 | 在这里的含义 |
| --- | --- |
| `x` | 目标 patch 的带噪版本 $ p_t $ |
| `t` | 噪声时间 |
| `c` | MAE 输出的条件特征 $ z_{p\mid B} $ 或 $ z_{p\mid AB} $ |


MLP 的输出与干净目标 $ p $ 比较，因此这里实现的是 clean-target prediction。[源码：MLP 实现](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L451-L468)

这也解释了为什么图中还要在 MAE 后面放一个 MLP：

> **MAE 将可见上下文转成目标位置的条件；MLP 结合这个条件和当前带噪目标，完成具体的去噪预测。**
>

对同一组上下文，MAE 的条件特征可以重复使用，而 MLP 可以在多个噪声时间上反复评估。

## 6. 两种条件的预测误差，如何变成上方的监督？
代码对 $ B $、$ AB $ 使用相同的目标、相同的时间、相同的噪声，只改变条件特征。

这样比较的就是：

$ \text{仅使用 }B\text{ 的预测误差}
-
\text{使用 }A\cup B\text{ 的预测误差}. $

实际实现中的估计为：

$ \widehat{s}(A,p\mid B)
=
\frac{1}{V}
\sum_{v=1}^{V}
g(t_v)
\left[
\|\hat p_{B,v}-p\|_2^2
-
\|\hat p_{AB,v}-p\|_2^2
\right], $

其中：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/58315253/1794469323255-f1690f13-d959-4c0a-abe8-38d653f4a647.png)

$ g(t)
=
\frac{t}{2\max(1-t,\,0.33)^2}. $

这对应源码：

```python
freq_weight = 1/2*t_bp_vpr/(1-t_bp_vpr).clamp_min(0.33).pow(2)

h_p_vprb = (
    freq_weight
    * (x_bp_vpr_pred - x_bp_vpr).pow(2).sum(-1, True)
)
```

随后：

```python
pmi_target = (
    h_p_given_B_preintegration
    - h_p_given_BA_preintegration
).mean(-1)
```

[源码：加权误差](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1047-L1063)、[源码：误差相减](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1109-L1114)。

### 用一个数值例子理解符号方向
假设某个时间、某次噪声下：

$ \|\hat p_B-p\|^2=10,
\qquad
\|\hat p_{AB}-p\|^2=3. $

则这一项贡献：

$ g(t)(10-3)=7g(t)>0. $

含义是：加入 $ A $ 后，模型对目标 $ p $ 的预测改善了。

上方的表征分支就应该给出相应的正分数：

$ \langle R(A),R(Bp)-R(B)\rangle>0. $

这只是帮助理解的数值例子，实际目标会对多个时间和噪声取平均。README 中设置 `diff_repeat_repr=32`，即 $ V=32 $；**这 32 次是用于估计目标的 Monte Carlo 评估，不是连续执行 32 步图像生成。**[源码：README 训练配置](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/README.md#L23-L35)

另外，真实实现包含时间加权、截断和有限次采样。因此这里应称为**用于训练的加权 PMI 估计**，不能把它直接当成精确计算出来的普通条件概率对数比。

## 7. 上方的 $ R(A) $、$ R(B) $、$ R(Bp) $ 在代码中怎样生成？
对于仓库提供的模型配置，空间表征路径是：

$ \text{视图}
\xrightarrow{E:\,18\text{ 层}}
\text{上下文 token}
\xrightarrow{\text{3 层 self-attention}}
\text{进一步处理的上下文}
\xrightarrow{\text{3 层 cross-attention}}
\text{空间表征}. $

生成分支的 MAE decoder 则是另一个 6 层模块。[源码：模型配置](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1219-L1237)、[源码：表征头](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1065-L1091)。

### Figure 1 右侧的 cross-attention，该怎样准确理解？
第一层 cross-attention 中：

+ **Query**：由“目标输出位置相对当前可见区域中心的偏移”索引得到的可学习 embedding；
+ **Key、Value**：共享 $ E $ 的输出，再经过那 3 层额外 self-attention 后得到的上下文特征。

源码的调用关系是：

```python
repr_tokens = repr_factor_pos_pad

for block in self.representation_encoder:
    repr_tokens = block(
        repr_tokens, z_e, q_rope, k_rope,
        attn_mask=attn_mask
    )
```

其中 `repr_tokens` 提供 query，`z_e` 提供 context。[源码：表征 cross-attention 调用](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1073-L1087)、[源码：Q/K/V 投影](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L224-L237)。

这里也需要把上一条的简化描述收紧：

> **“Query 是位置”描述的是最初输入。经过第一层 cross-attention 后，query token 已经读入图像内容；后续层继续更新它。**
>

### 7.1 把 self-attention、cross-attention 和 $ H $ 写成张量形状
为避免把“空间输出位置的 $ Q $”和“attention 里的 Query $ Q $”混在一起，下面把空间输出位置数记为 $ N_q $。对一个视图，encoder 输出可抽象为：

$ z_e\in\mathbb{R}^{B_v\times K\times d}, $

其中 $ B_v $ 是打包后的视图 batch，$ K $ 是 register/context token 加上被选入 encoder 的 patch token 数，$ d=\texttt{hidden\_size} $。以常见的 REPR-B/16 配置为例，$ d=768 $、attention head 数 $ h=12 $，每个 head 的维度为：

$ d_h=\frac{d}{h}=64. $

#### `repr_encoder` 的 self-attention
`repr_encoder` 的三层 self-attention 处理的是同一条 encoder context 序列。输入为 $ [B_v,K,d] $；线性投影并拆成多头后：

$ Q,K,V\in\mathbb{R}^{B_v\times h\times K\times d_h}. $

attention 输出重新合并 head 后仍为 $ [B_v,K,d] $。它的作用是让可见 patch token、register token 在表征路线内部互相交换信息，但不会改变序列长度 $ K $。

#### `representation_encoder` 的 cross-attention
cross-attention 的 query 不是 encoder 序列本身，而是一个带 padding 的空间查询网格：

$ x=\texttt{repr\_tokens}\in\mathbb{R}^{B_v\times N_q\times d}. $

对于 $ 256\times256 $ 输入和 $ 16\times16 $ patch，原始网格是 $ 16\times16 $；代码向四周 padding 一格，所以常见的空间查询数是：

$ N_q=18\times18=324. $

每层 cross-attention 中：

$ \begin{aligned}
Q&=W_qx\in\mathbb{R}^{B_v\times h\times N_q\times d_h},\\
K&=W_kz_e\in\mathbb{R}^{B_v\times h\times K\times d_h},\\
V&=W_vz_e\in\mathbb{R}^{B_v\times h\times K\times d_h}.
\end{aligned} $

注意力结果回到 $ [B_v,N_q,d] $，再经过残差和 feed-forward block。三层 cross-attention 复用同一个空间查询序列逐层更新：初始 `repr_tokens` 主要是学习到的相对位置 embedding；第一层从 $ z_e $ 读取图像内容，后两层继续细化。它不是“只把坐标送入最后一层”，而是一个逐层写入上下文的空间表征解码过程。

`attn_mask` 会限制 query 能读取的 context，主要保留 register token 和真实可见 token，避免补齐用的 masked/filler token 泄漏信息。

因此可以把代码中的 $ H $ 展开为：

$ H_\theta
=
\texttt{repr\_encoder}_{\text{3 self-attn}}
\;\rightarrow\;
\texttt{representation\_encoder}_{\text{3 cross-attn}}
\;\rightarrow\;
\texttt{repr\_final\_layer}. $

空间输出为 $ [B_v,N_q,\texttt{repr\_dim}] $。全局表征走另一条短路径：取 `repr_encoder` 输出中的 in-context/register token，经过 `repr_final_layer_global`，不再经过空间 cross-attention。也就是说，$ H $ 不是一个独立预训练的单层 head，而是“3 层 self-attention + 3 层 cross-attention + 输出投影”的完整表征头。

[源码：表征头构造、self-attention 与 cross-attention](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L648-L661)

### $ R(Bp)-R(B) $ 是整张表征图的变化
第一轮中，$ Bp $ 分支确实能看到真实的 $ p $，而且代码会在新增的 $ p $ 上加一个特殊的可学习标记 `self.is_b_embed`。这样模型知道哪个 patch 是本次新增的目标。[源码：](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L978-L988)$ Bp $[ 内容与角色标记](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L978-L988)

随后，$ B $、$ Bp $ 分别生成完整空间表征，再相减：

$ R(Bp)-R(B). $

新增一个输入 patch，可能改变所有输出位置的特征；代码会对整张表征图计算点积。

忽略无效样本筛选、padding dropout 等处理，实际空间分数的结构是：

$ s_{\mathrm{repr}}
=
\frac{\sqrt{d_E}}{2d_R\sqrt Q}
\sum_{q=1}^{Q}
\left\langle
R_q(A),\,
R_q(Bp)-R_q(B)
\right\rangle, $

其中 $ d_E $ 是 encoder hidden size，$ d_R $ 是输出表征维度，$ Q $ 是空间输出位置数量。[源码：空间分数](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1088-L1114)

对于 README 的 $ 256\times256 $ 输入、$ 16\times16 $ patch 配置，原图有 $ 16\times16 $ 个位置；训练空间输出还向四周各扩展一格，因此通常是：

$ Q=18\times18=324. $

这也比“输出仅有原图的 $ 16\times16 $ 网格”的说法更准确。[源码：空间 padding 定义](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L560-L570)

## 8. 两条分支怎样一起训练？哪些参数接收梯度？
Figure 1 中的 `Stopgrad`，在代码中对应这一段：

```python
with torch.no_grad():
    # 计算两个条件下的去噪误差
    # 构造用于表征学习的监督目标
    ...
```

因此，RePR loss 不会通过这个监督目标反向传播到 decoder 和 diffusion MLP。但生成分支还有自己的 MAR loss，照常更新参数。

联合训练步的梯度关系是：

| 参数模块 | MAR loss 更新 | RePR loss 更新 |
| --- | ---: | ---: |
| 共享 encoder $ E $ | ✓ | ✓ |
| 生成 decoder $ D $ | ✓ | — |
| Diffusion MLP $ M $ | ✓ | — |
| 表征头 $ H $ | — | ✓ |


[源码：MAR loss 与停止梯度的目标计算](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1033-L1063)、[源码：表征计算与总损失](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1065-L1120)。

真实训练实现还有三点需要补全：

1. **联合 `forward_repr()` 中，MAR loss 同时训练 $ B $ 和 $ AB $ 两种条件。**我上一条按论文简化伪代码只讲了 $ AB $，对这个仓库来说不完整。单独的 `forward_mar()` 则使用 $ AB $ 条件。[联合步源码](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1010-L1045)、[纯 MAR 步源码](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L884-L910)
2. **实际 loss 用的是 Huber。**空间表征之外，代码还从 register token 构造全局表征，并加入一项 global RePR loss：

$ L
=
L_{\mathrm{MAR}}
+\lambda L_{\mathrm{RePR,spatial}}
+\lambda L_{\mathrm{RePR,global}}. $

3. 图中的 L2 是更简化的描述；代码使用 `2 * huber_loss`，在小残差区间与平方误差一致，大残差时改为线性惩罚。[源码：损失定义](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1116-L1120)
4. **每 4 步中有 1 步做联合训练，其余 3 步只做 MAR。**具体判断是 `step % 4 == 0` 时调用 `forward_repr()`。[源码：训练调度](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/denoiser_repr.py#L91-L100)

### 8.1 Huber loss 到底做了什么？
代码里的变量名仍然叫 `mse_loss`，但实际调用的是 `2 * F.huber_loss`。对残差 $ e=\hat y-y $，PyTorch 的 Huber 定义为：

$ \operatorname{Huber}_\delta(e)=
\begin{cases}
\frac12e^2,&|e|\le\delta,\\
\delta\left(|e|-\frac12\delta\right),&|e|>\delta.
\end{cases} $

代码乘以 2 后变成：

$ 2\operatorname{Huber}_\delta(e)=
\begin{cases}
e^2,&|e|\le\delta,\\
2\delta|e|-\delta^2,&|e|>\delta.
\end{cases} $

所以小误差区间与 MSE 完全一致，大误差区间改为线性惩罚，梯度不再随 $ |e| $ 无限增大。这能避免少数异常 patch、极端噪声时间或无效候选对训练产生过大的梯度。

在 MAR 项中，代码使用 `delta=10`，并且还保留时间权重与有效 mask 归一化：

$ L_{\mathrm{MAR}}
\propto
\frac{2\,\operatorname{Huber}_{10}(\hat p-p)}
{\max(1-t,0.33)^2}. $

这里的 PMI target 本身是在 `torch.no_grad()` 中由两种条件的**平方预测误差之差**构造的；Huber 作用在最终的 MAR/RePR 训练损失上，不是把内部的 PMI 误差估计改成 Huber。

空间与全局 RePR 回归项也使用 `2 * huber_loss`，但阈值是 `repr_delta`（README 常见配置为 $ 0.2 $）：

$ L
=
L_{\mathrm{MAR}}
+\lambda L_{\mathrm{RePR,spatial}}
+\lambda L_{\mathrm{RePR,global}}. $

### 8.2 “每 4 步只优化一次 MAR”是怎样的误读？
不是每四步才优化一次 MAR。训练循环每个 batch 都执行一次 `zero_grad()`、`backward()` 和 `optimizer.step()`；代码没有因为四步调度而累积梯度。

`denoiser_repr.py` 的调度是：

| step 条件 | 调用 | 当步损失 | MAR 是否更新 | RePR 是否更新 |
| --- | --- | --- | ---: | ---: |
| `step % 4 == 0` | `forward_repr()` | MAR + spatial RePR + global RePR | ✓ | ✓ |
| 其余三步 | `forward_mar()` | MAR-only | ✓ | — |


所以 step $ 0 $ 做一次联合步，step $ 1,2,3 $ 做三次 MAR-only，step $ 4 $ 再做联合步。MAR 在四步中每一步都被优化；RePR 只在每四步中的一个 step 计算和反向传播。

联合步中，MAR 仍然会比较 $ B $ 与 $ AB $ 两类条件，并训练共享 encoder、decoder 和 diffusion MLP。RePR 的 PMI target 构造放在 `no_grad()` 中，只切断“用预测误差生成监督目标”的路径；预测侧的共享 encoder 和表征头仍会从 RePR loss 获得梯度，decoder 与 diffusion MLP 则从 MAR loss 获得梯度。

[源码：每四步调度](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/denoiser_repr.py#L132-L181)、[源码：Huber 与总损失](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1213-L1258)

### 训练完成后，提取表征走哪条路径？
仓库的 `forward_feats()` 调用的是：

$ X\rightarrow E_\theta\rightarrow H_\theta\rightarrow R_\theta(X). $

这个函数没有调用生成 decoder $ D $ 或 diffusion MLP $ M $。生成分支在训练时提供预测任务和 PMI 监督；提取表征时，保留的是共享 encoder 加表征头。[源码：](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1125-L1216)`forward_feats()`

第二轮训练沿用这套结构，但先用冻结的上一轮模型产生 feature target，去噪目标也随之从像素 patch 换成对应位置的特征；$ Bp $ 中新增目标的嵌入也会走 `feats_embedder`。[源码：第二轮目标生成](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/denoiser_repr.py#L102-L118)、[源码：目标嵌入选择](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L928-L933)

## 9. 第一阶段与第二阶段的交接
本文前面的主体可以限定为第一阶段，也就是 `is_first_iter=True` 的像素目标训练。第一阶段每个联合 step 的目标是：

$ L
=
L_{\mathrm{MAR,pixel}}
+\lambda L_{\mathrm{RePR,spatial}}
+\lambda L_{\mathrm{RePR,global}}. $

其中，RePR 让 $ E+H $ 的表征点积去拟合“加入 $ A $ 后，目标 $ p $ 的预测误差下降多少”；MAR 则让共享 encoder 保持对图像结构的可预测性。因而第一阶段不是“只训练表征、生成分支只是摆设”：$ E+D+M $ 仍然在像素 patch 上进行去噪预测，只是最终导出的表征路径不需要 $ D $ 和 $ M $。

训练完成后，`forward_feats()` 的路径是：

$ X
\xrightarrow{\text{patchify + x\_embedder}}
E_\theta
\xrightarrow{\text{3 self-attn + 3 cross-attn + final layer}}
H_\theta
\longrightarrow R_\theta(X). $

它不调用 MAE decoder，也不调用 diffusion MLP。这个 $ E+H $ 的输出才是第二阶段 teacher 提供的空间 feature target。

第二阶段 `repr_mar_featspred` 的交接过程是：

1. 从第一阶段 checkpoint 构造冻结的 `frozen_net`；它不是训练中的 EMA，而是一个不会更新参数的上一阶段 teacher。
2. teacher 调用 `forward_feats(x, return_padded=False)`，得到原图 patch 位置上的空间表征 $ x_{\mathrm{target}}\in\mathbb{R}^{B\times N\times\texttt{repr\_dim}} $；包装层只取空间分支，global 返回值不作为这个 target。
3. student 仍然从当前 RGB 图像构造 $ A $、$ B $ 上下文，并保留 encoder、MAE decoder 和 diffusion MLP。区别是 $ B_p $ 新增的目标以及 MAR 的去噪目标都切换到 teacher feature space，`diff_mlp` 预测的是 `repr_dim` 维 latent feature。
4. 第二阶段仍按每四步一次 `forward_repr()`、三步 `forward_mar()` 的调度训练，因此它仍然是 feature-space MAR/RePR，而不是纯粹的 encoder-only 训练。

所以“第二轮只取 encoder 出来的特征”需要改成更准确的表述：**第二轮 teacher 的 target 来自冻结模型的 **$ E+H $** 表征路径，而不是裸 **$ E $**；第二轮 student 的训练仍使用 **$ E+D+M $** 预测这些 feature target。最终部署或 tokenizer 提特征时再次使用 **$ E+H $**，只有需要继续做生成式 feature prediction 时才保留 **$ D+M $**。**

当前仓库代码能够直接证实的是上述 RePR 训练、冻结 teacher 和 `forward_feats()` 导出路径；后续 tokenizer 或其它生成器如何消费这些 feature，需要查看后续项目代码，不能仅凭当前仓库断言其具体实现。

[源码：第二阶段 teacher 与 target 生成](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/denoiser_repr.py#L151-L181)、[源码：](https://github.com/zrbwsw/repr/blob/cb1c132c7c3d2e0ee9c21313c8e2ec82f6127cc3/model_repr.py#L1345-L1451)`forward_feats()`

第一阶段到第二阶段可以概括为：

$ \text{第一阶段：像素 patch MAR + RePR}
\quad\Longrightarrow\quad
\text{冻结 }E+H\text{ 产生 feature target}
\quad\Longrightarrow\quad
\text{第二阶段：feature-space MAR + RePR}. $

