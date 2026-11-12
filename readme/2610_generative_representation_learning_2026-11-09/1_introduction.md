## 1.1 Representation learning 决定哪些信息值得保留
论文开头提出一个基本观点：

> Representation learning 决定了数据中什么信息是重要的。
>

传统表示学习通常围绕下游任务设计。例如，分类任务希望 representation：

+ 保留物体类别和语义；
+ 对光照变化保持不变；
+ 对背景变化保持一定不变性；
+ 对裁剪、颜色变化和局部扰动具有鲁棒性。

因此，分类 representation 通常会主动丢弃一些与类别无关的变量。

但近年来，这些判别式 representation 被大量用于生成模型 tokenizer。例如：

$ x \xrightarrow{E_\theta} z \xrightarrow{D_\phi} \hat{x}. $

其中：

+ $ E_\theta $ 是 encoder；
+ $ z $ 是 latent tokens；
+ $ D_\phi $ 是 decoder。

完整的生成链路还包括 latent generative model：

$ x \xrightarrow{E_\theta} z,
\qquad
\tilde z\sim G_\psi,
\qquad
\tilde x\sim D_\phi(\tilde z). $

生成 tokenizer 不能只保留物体类别，还需要保留：

+ 形状与布局；
+ 背景与场景结构；
+ 纹理与材质；
+ 颜色和局部细节；
+ 高频信息以及其他会影响图像生成的变量。

因此，适合分类的 representation 不一定适合生成。

## 1.2 RAE 和判别式 encoder 带来的问题
论文以 RAE 一类方法为背景。RAE 使用预训练 representation encoder，例如 DINOv2，然后训练 decoder 将 latent 恢复成图像。

这种做法的优点是可以利用强大的预训练语义表示，但它也存在两个问题：

1. DINOv2 的训练目标原本不是为了生成设计；
2. encoder 的预训练成本可能比后续 generative model 还高很多。

这使论文提出两个基本问题。

### 问题一：什么样的 representation 对生成有利？
生成 representation 需要保留所有对生成有用的变量，而不是只保留少量语义变量。

### 问题二：训练 tokenizer 是否真的比直接 pixel modeling 更省计算？
需要比较：

$ \text{Total FLOPs}
=
\text{Tokenizer Encoder FLOPs}
+
\text{Decoder FLOPs}
+
\text{Generative Model FLOPs}. $

如果 encoder 预训练本身极其昂贵，那么 tokenizer 是否具有计算优势就不能只看最终 FID，还必须进行端到端 FLOP 比较。

## 1.3 过去的 tokenizer scaling 可能混合了 decoder 和 encoder 的作用
论文指出，已有工作已经观察到 tokenizer 的 scaling law，但很多工作将 tokenizer decoder 与 tokenizer encoder 绑定在一起。

decoder 属于生成过程的一部分，类似 diffusion model 或 autoregressive model，需要学习条件分布：

$ p_\phi(x\mid z). $

生成模型通常随着参数量、数据量和训练计算量增加而改善。

然而，encoder 的 scaling 可能完全不同。已有工作观察到，一些 encoder 在规模扩大后，其 patch features 可能出现退化、不稳定或更难用于生成。

因此，整体 tokenizer 的性能变好并不能直接证明 encoder 本身变好了。

可以将完整系统写成：

$ Q=Q(E,D,G), $

其中 $ E $ 是 encoder，$ D $ 是 decoder，$ G $ 是 latent 或 pixel generative model。

如果增大 $ D $ 和 $ G $ 能显著提升 $ Q $，即使增大 $ E $ 的收益很小，甚至造成一定退化，整体结果仍可能变好。

论文真正希望研究的是：

> 在固定后续生成模型的条件下，encoder 的训练计算量和模型规模增加，是否能够让 latent 更容易被生成模型建模？
>

## 1.4 现有 encoder 可能失败的三个原因
### 1.4.1 Feature suppression
对比学习和 LeJEPA 等方法可能只学习完成判别目标所需的少量信息。

假设图像由两类变量组成：

$ x=(s,n), $

其中：

+ $ s $ 是类别和身份等语义变量；
+ $ n $ 是纹理、背景、局部结构等生成变量。

分类任务可能只需要：

$ z\approx s, $

而生成任务需要：

$ z\approx (s,n). $

如果一个 watermark 已经足以区分样本，判别目标可能被 shortcut 满足，而不再要求模型保留真实图像内容。

对于生成 tokenizer，这会导致生成所需的变量被压制。

### 1.4.2 Causal/theoretical grounding 不足
对比学习和 DINO 通常主要监督 CLS token，但 RAE 等生成 tokenizer 实际使用的是 encoder 的内部 patch features。

因此会出现：

+ CLS token 的训练目标清晰；
+ patch tokens 只是内部中间变量；
+ patch features 没有被直接要求具有稳定的生成结构；
+ 模型规模增加后，patch features 可能出现 outlier、漂移和各向异性。

论文认为，生成 tokenizer 应该直接监督实际输出的 patch representations。

### 1.4.3 Architectural scalability
经典 VAE 和 VQ-VAE 使用 KL 正则、离散 codebook、reparameterization trick 或 straight-through estimator。

这些结构可能带来：

+ 梯度传播瓶颈；
+ codebook 学习困难；
+ 语义表示不足；
+ 生成和重建目标之间的折中。

因此，现代方法转向使用 DINO 等预训练 representation encoder，但又继承了判别式目标与生成目标不一致的问题。

## 1.5 从判别式 representation 转向生成式 representation
论文提出将 representation learning 从 discriminative task 转向 generative task。

作者注意到，Word2Vec 和 contrastive learning 都可以与 PMI factorization 联系起来。传统对比学习通常通过 negative samples 间接估计 PMI，而论文希望使用生成模型直接估计 PMI。

基本公式为：

$ \operatorname{PMI}(x,y)
=
\log p(y\mid x)-\log p(y). $

然后让 representation 的内积匹配 PMI：

$ \langle R_\theta(x),R_\theta(y)\rangle
\approx
\operatorname{PMI}(x,y). $

这种方法有几个目标：

+ 保留对比学习的 PMI 理论基础；
+ 不使用 negative samples；
+ 避免 hard negative 问题；
+ 减轻 feature suppression；
+ 直接监督 patch-level representation；
+ 利用生成模型已有的 scaling law；
+ 使用标准 Transformer 和连续梯度下降。

## 1.6 实验策略
论文在 ImageNet 和 GPIC 上研究生成模型 tokenizer。

作者使用 Latent Forcing 来隔离 encoder 的 diffusability：

1. 先在 latent 空间中进行去噪；
2. 再直接生成像素；
3. 避免使用传统有损 decoder 作为主要性能来源；
4. 固定下游生成模型的规模和训练时长；
5. 观察 encoder 规模和训练 FLOPs 对生成质量的影响。

论文还研究：

+ unconditional generation；
+ conditional generation；
+ guided generation；
+ unguided generation；
+ GAN+LPIPS decoding。

作者声称，RePR 的收益主要来自 representation 的 diffusability，而不是 perceptual loss、adversarial loss 或外部 decoder。

## 1.7 Introduction 中列出的论文贡献
### 贡献一：提出 RePR
RePR 是一种具有理论基础的 representation learning 算法，用于发现不同 masked views 之间共享的 latent variables。

### 贡献二：在高维真实图像上实现
论文不仅在理论上讨论 PMI factorization，也给出了在 ImageNet 和 GPIC 上的实际实现。

### 贡献三：分析 tokenizer encoder scaling law 和总 FLOP 效率
论文试图证明：

+ RePR encoder 可以随规模和训练计算量改善；
+ representation learning 可以作为生成模型之外的额外 scaling axis；
+ 在总 FLOPs 相近时，RePR + Latent Forcing 可以优于直接 pixel diffusion。

## 1.8 Introduction 的核心结论
Introduction 的核心逻辑可以总结为：

1. 判别式 representation 被广泛用作生成 tokenizer，但它们的训练目标并非为生成设计。
2. 传统方法可能出现 feature suppression，丢弃生成所需的信息。
3. CLS token 的监督不能保证内部 patch features 稳定、可解释和易于生成建模。
4. 过去的 tokenizer scaling 可能混合了 decoder 的收益，因此不能直接证明 encoder scaling。
5. RePR 使用生成模型估计 PMI，并直接监督 patch representation。
6. 论文希望证明生成式 representation learning 能够产生真正的 tokenizer encoder scaling law。

