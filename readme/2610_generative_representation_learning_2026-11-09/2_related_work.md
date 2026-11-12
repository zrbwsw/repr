本章对应论文原文的三个小节：

1. Representation Learning
2. Pointwise Mutual Information and Representations
3. Pixel and Latent Space Generative Modeling

这三个小节共同回答一个问题：RePR 继承了哪些已有思想，又试图解决哪些已有方法没有解决的问题？

# 2.1 Representation Learning
## 2.1.1 这一小节的作用
论文首先回顾自监督 representation learning 的两条主要路线：

1. contrastive/discriminative learning；
2. predictive/generative learning。

作者并不是否定这些方法，而是指出：它们的训练目标可能适合分类和语义任务，却不一定适合生成 tokenizer。

生成 tokenizer 需要的 representation 不只是“能够区分样本”，还需要保留对生成有用的全部变量，并且让这些变量在 latent 空间中具有良好的可建模性。

## 2.1.2 对比学习：基本思想
对比学习构造正样本对和负样本。

对于同一图像的两个增强视图 $ x_i $ 和 $ x_i^+ $，它们是 positive pair；其他图像的视图是 negative samples。

InfoNCE 可以写成：

$ \mathcal{L}_{\mathrm{InfoNCE}}
=
-\log
\frac{
\exp(\operatorname{sim}(z_i,z_i^+)/\tau)
}{
\sum_j \exp(\operatorname{sim}(z_i,z_j)/\tau)
}. $

其目标是：

+ 拉近同一实例的不同 view；
+ 推远不同实例的 view；
+ 学习能够区分样本的 representation。

典型方法包括：

+ SimCLR；
+ MoCo；
+ CPC；
+ DINO 系列；
+ 其他 instance discrimination 方法。

DINO 不完全等同于标准 InfoNCE，因为它主要采用 teacher-student self-distillation、centering 和 sharpening，但论文将它与对比式方法放在同一类问题中讨论：它们都主要面向判别性和语义表示。

## 2.1.3 对比学习的问题一：Hard Negatives
Hard negative 是与 positive 在语义上很相似、但被训练为负样本的样本。

例如：

+ 两张同一类别的图像；
+ 两个相似物体的不同实例；
+ 同一个人的不同照片；
+ 同一场景的相邻裁剪；
+ 外观和布局都非常接近的图像。

对比学习希望把这些样本推远，但它们实际上共享很多信息。为了区分它们，模型可能编码：

+ 背景；
+ 光照；
+ 纹理；
+ 拍摄角度；
+ 实例特有细节；
+ 水印或压缩噪声。

这会产生两个问题。

### 问题一：训练信号不稳定
如果 negative 太难，模型需要很大容量才能区分它们；如果 negative 采样不合理，梯度方向可能与真正想要的语义结构不一致。

### 问题二：模型可能学习不希望保留的差异
为了区分两个语义非常相近的样本，模型可能依赖背景、纹理或水印。

这对实例识别可能有帮助，但对生成 tokenizer 来说可能有害，因为生成模型需要保留完整图像变量，而不是只学习区分样本的 shortcut。

## 2.1.4 强条件视图为什么会加剧 Hard Negative 问题？
论文提到，contrastive learning 在 strongly conditioned views 下容易遇到 hard negative 问题。

如果两个 view 之间共享大量信息，例如：

+ 同一个物体；
+ 相似布局；
+ 相同实例身份；
+ 相似背景；

那么要找到真正合适的 negative sample 并不容易。

如果 negative 太相似，模型需要编码非常细的差异；如果 negative 太简单，InfoNCE 可能很快饱和，训练信号变弱。

因此，对比学习通常需要：

+ 更大的 batch size；
+ memory queue；
+ hard-negative mining；
+ 更复杂的采样策略；
+ 精心设计的数据增强。

RePR 的思路是绕过这个问题，不再依赖大量 negative samples，而是直接比较条件预测和无条件预测：

$ \operatorname{PMI}(x,y)
=
\log p(y\mid x)-\log p(y). $

它问的是：

> 给定 $ x $ 后，预测 $ y $ 的难度降低了多少？
>

而不是：

> $ x $ 是否比其他样本更加相似？
>

## 2.1.5 对比学习的问题二：Feature Suppression
Feature suppression 表示 representation 没有完全塌缩，但主动丢弃了很多不影响当前训练目标的变量。

需要区分：

### Dimensional collapse
不同样本的 representation 几乎相同，或者所有 representation 只占据很低维的子空间。

### Feature suppression
representation 仍然可以区分样本，但它压制了对当前任务不必要的变量。

假设图像包含：

$ x=(s,n), $

其中：

+ $ s $ 是语义和类别变量；
+ $ n $ 是纹理、背景、局部结构和其他细节。

对于分类任务，模型可能只需要：

$ z\approx s, $

对于生成任务，则需要：

$ z\approx (s,n). $

如果 $ s $ 已经足以完成实例区分，那么对比目标没有动力要求模型保留 $ n $。

因此，模型可能出现：

+ 语义表示很强；
+ 分类准确率很高；
+ representation 没有完全 collapse；
+ 但纹理、背景和场景结构被压制。

这正是生成 tokenizer 需要避免的问题。

## 2.1.6 Watermark 实验如何说明 Feature Suppression？
论文给 ImageNet 图像加入一个唯一 watermark，使每张图像都可以被 watermark 区分。

在这种情况下，判别式方法只要学习 watermark，就可以完成实例识别目标。

因此：

+ DINOv3 和 LeJEPA 可能主要学习 watermark；
+ 图像的真实内容不再是完成任务所必需的；
+ 表示可能失去大量生成相关信息；
+ RePR 更能保留图像本身的变量。

这个实验说明：

> 判别任务完成得很好，不等于 representation 保留了生成任务需要的信息。
>

---

## 2.1.7 数据增强与 Feature Suppression
为了缓解对比学习的问题，许多方法使用 hand-crafted augmentations：

+ Random Crop；
+ Color Jitter；
+ Gaussian Blur；
+ Grayscale；
+ Horizontal Flip；
+ Multi-crop。

如果一种变化在正样本的两个 view 中经常被改变，模型就会被鼓励忽略它。

例如：

+ 强 Color Jitter 会减少模型对颜色的依赖；
+ 强 Random Crop 会减少模型对绝对位置的依赖；
+ 强 Blur 会减少模型对高频纹理的依赖；
+ Grayscale 会减少模型对颜色信息的依赖。

这对分类可能是有益的，但生成模型可能需要这些信息。

因此，数据增强实际上决定了：

> 哪些变量被视为应该保持不变，哪些变量会被模型逐渐忽略。
>

对于生成 tokenizer，需要谨慎选择增强策略，否则增强本身可能主动删除生成所需的信息。

## 2.1.8 Predictive Objectives：预测式表示学习
论文随后介绍预测式表示学习。

这类方法不是显式地拉近 positive、推远 negative，而是让模型根据可见上下文预测缺失内容。

主要包括两种形式：

1. 预测原始像素；
2. 预测神经网络 latent。

## 2.1.9 Pixel Prediction：预测像素
代表方法包括：

+ Context Encoders；
+ MAE；
+ masked image modeling。

给定图像可见部分 $ x_{\mathrm{visible}} $，模型预测被 mask 的区域 $ x_{\mathrm{masked}} $：$ \mathcal{L}_{\mathrm{pixel}}
=
\left\|
x_{\mathrm{masked}}
-
\hat{x}_{\mathrm{masked}}
\right\|_2^2. $

优点包括：

+ 不需要 negative samples；
+ 直接监督局部图像信息；
+ 对 patch-level 表示有明确约束；
+ 比只监督 CLS 更接近 tokenizer 使用方式。

但存在几个问题。

### 问题一：高频细节可能主导损失
像素级误差对高频纹理和局部变化很敏感，模型可能把大量容量用于拟合低层细节，而不是学习更有用的语义和结构。

### 问题二：像素预测目标可能存在多解
一个被 mask 的区域可能有多个合理内容，均方误差容易预测平均结果，造成模糊或缺乏结构的输出。

### 问题三：完整 likelihood 估计成本较高
对于大图像或复杂条件，准确估计整张图像的 likelihood 计算成本很大。

## 2.1.10 Latent Prediction：预测神经网络表示
另一类方法不是预测像素，而是预测 teacher 或 target network 的 latent。

代表方法包括：

+ BYOL；
+ Data2Vec；
+ JEPA；
+ MAGE；
+ 其他 teacher-student latent prediction 方法。

这类方法通常让 student 根据可见区域预测 target network 的 latent：

$ \mathcal{L}_{\mathrm{latent}}
=
\left\|
R_{\mathrm{student}}(x_{\mathrm{visible}})
-
\operatorname{stopgrad}
\left(
R_{\mathrm{teacher}}(x_{\mathrm{target}})
\right)
\right\|_2^2. $

相比像素预测，它们的目标更加抽象，对像素噪声不那么敏感，也不一定需要 negative samples。

但是它们可能继承 teacher representation 的信息选择：

+ 如果 teacher 只保留语义，student 也可能只学习语义；
+ 如果 teacher 发生 feature suppression，student 可能继承这种抑制；
+ 如果只监督全局 latent，patch-level representation 仍然可能缺乏结构约束；
+ teacher 的架构和训练方式会影响最终表示的内容。

因此，latent prediction 并不自动保证表示适合作为生成 tokenizer。

## 2.1.11 RePR 在 Representation Learning 中的位置
RePR 试图结合 contrastive learning 和 predictive learning 的优点。

### 与 contrastive learning 的联系
传统对比学习可以近似看作 PMI factorization：

$ \langle R_\theta(x),R_\theta(y)\rangle
\approx
\operatorname{PMI}(x,y)+a(x). $

RePR 不再通过 negative samples 间接估计 PMI，而是使用生成模型显式计算：

$ \operatorname{PMI}(x,y)
=
\log p(y\mid x)-\log p(y). $

因此，RePR 可以理解为：

> 将对比学习中隐式的 PMI 估计，改造成由生成模型提供的显式 PMI 监督。
>

### 与 pixel prediction 的联系
RePR 也使用 masked views 和局部预测，但它不直接复制像素，而是学习一个 view 对另一个 view 的预测增益。

因此，它关注的是信息量和可预测性，而不是单纯的像素误差。

### 与 latent prediction 的联系
RePR 也支持在 feature space 中进行迭代预测，但它并不是简单拟合一个预训练 teacher latent。

RePR 的 target 来自生成模型估计的 PMI，并且直接作用于 patch representations。

---

## 2.1.12 Representation Learning 小节的核心结论
这一小节的论证链可以总结为：

1. Contrastive learning 能够通过 view relationship 学到很强的语义 representation。
2. 但是 negative samples 会带来 <font style="background-color:#FBDE28;">hard negative</font> 问题。
3. 判别式目标可能只保留少量语义 bit，导致 <font style="background-color:#FBDE28;">feature suppression</font>。
4. 数据增强能够控制模型关注的变量，但也可能主动删除生成所需信息。
5. Pixel prediction 不依赖 negative samples，但可能被高频细节、像素多解和 likelihood 计算成本影响。
6. Latent prediction 的目标更抽象，但可能继承 teacher 的信息选择和 feature suppression。
7. RePR 使用生成模型估计 PMI，并直接监督 patch-level representation，试图让表示学习目标与生成任务对齐。

# 2.2 Pointwise Mutual Information and Representations
## 2.2.1 PMI 的定义
点互信息衡量两个事件共同出现的程度，相对于它们独立出现时的概率增加了多少。

定义为：

$ \operatorname{PMI}(x,y)
=
\log \frac{p(x,y)}{p(x)p(y)}. $

利用条件概率，可以写成：

$ \operatorname{PMI}(x,y)
=
\log \frac{p(y\mid x)}{p(y)}
=
\log p(y\mid x)-\log p(y). $

如果 $ x $ 的存在大幅提高了 $ y $ 的概率，那么 PMI 较高。

如果 $ x $ 和 $ y $ 近似独立，那么：

$ p(y\mid x)\approx p(y), $

因此：

$ \operatorname{PMI}(x,y)\approx 0. $

如果 $ x $ 的存在反而使 $ y $ 更不可能，则 PMI 可以为负。

## 2.2.2 Word2Vec 与 PMI factorization
论文指出，Levy 和 Goldberg 证明了 Word2Vec 的词向量内积可以看作 PMI 矩阵的低秩 factorization。

直观地说，词向量之间的内积反映两个词在语料中共同出现的统计关系。

因此，学习 representation 可以被理解为：

> 找到一个低维向量空间，使向量之间的内积近似描述数据中的统计依赖关系。
>

## 2.2.3 Contrastive Learning 与 PMI
论文进一步指出，InfoNCE 等对比学习方法也可以与 PMI factorization 联系起来：

$ \langle R_\theta(x),R_\theta(y)\rangle
\approx
\operatorname{PMI}(x,y)+a(x). $

这解释了为什么对比学习能够学习有意义的 representation：它实际上在估计不同 view 或样本之间的统计依赖。

但是，传统对比学习通常依赖 negative samples 来估计这种关系。

当 negative sampling 困难、view 条件很强或存在 shortcut 时，PMI 的估计可能不稳定，representation 也可能出现 feature suppression。

## 2.2.4 Platonic Representation Hypothesis
论文还提到 Platonic Representation Hypothesis。

这一假设认为，不同的大模型在规模足够大时，可能会逐渐收敛到相似的 representation structure。

PMI factorization 被视为这种共同 representation space 的一个候选解释。

但是，论文同时指出，已有研究发现，大规模神经网络的 representation 并不一定继续稳定地匹配这种结构。

因此，不能简单假设：

> 模型规模变大后，好的 representation 会自动涌现。
>

RePR 的做法是直接监督 representation 去匹配 PMI，而不是仅仅等待 PMI 结构在大模型内部自然 emergence。

## 2.2.5 RePR 如何使用 PMI
RePR 使用生成模型直接估计 PMI：

$ \operatorname{PMI}(x,y)
=
\log p(y\mid x)-\log p(y). $

然后让 representation 内积匹配这个 PMI：

$ \mathcal{L}_{\mathrm{PMI}}
=
\left(
\langle R_\theta(x),R_\theta(y)\rangle
-
\operatorname{PMI}(x,y)
\right)^2. $

相比传统对比学习，RePR 的重要区别是：

+ 不需要显式构造大量 negative samples；
+ 条件 likelihood 和无条件 likelihood 都由生成模型估计；
+ 可以通过 timestep weighting 控制不同信息尺度；
+ 可以直接作用于 patch features；
+ 可以利用生成模型本身的 scaling law。

## 2.2.6 为什么 PMI 对生成 tokenizer 有意义？
生成 tokenizer 需要保留不同 view 之间共享并且可预测的信息。

如果一个 patch view 能够帮助预测另一个 patch view，那么这部分信息对生成模型有价值。

PMI 正好描述了这种预测增益：

$ \text{prediction gain}
=
\log p(y\mid x)-\log p(y). $

因此，RePR 的表示不是简单地学习“两个 view 看起来相似”，而是学习：

> 一个 view 为预测另一个 view 提供了多少信息。
>

这比单纯的实例级相似度更加接近生成任务。

# 2.3 Pixel and Latent Space Generative Modeling
## 2.3.1 Diffusion Transformer 的两种生成空间
论文将现代图像生成方法分为两类：

1. pixel-space diffusion；
2. latent-space diffusion。

### Pixel-space diffusion
模型直接在像素空间中进行 diffusion 或 flow generation：

$ x_0\rightarrow x_t\rightarrow x_1. $

代表方法是 JiT。

优点是：

+ 不需要额外 tokenizer；
+ 不存在有损 latent compression；
+ 生成过程直接作用于图像。

缺点是：

+ 图像维度很高；
+ token 数量较多；
+ 生成模型计算成本较大；
+ 对高分辨率图像尤其困难。

### Latent-space diffusion
latent diffusion 先使用 tokenizer 将图像映射到 latent：

$ x\xrightarrow{E}z, $

然后在 latent 空间中训练生成模型：

$ z\sim p_\psi(z), $

最后使用 decoder 恢复图像：

$ \hat{x}=D(z). $

它的目标是降低后续生成模型的空间复杂度和计算量。

---

## 2.3.2 Latent diffusion 的两阶段结构
Latent diffusion 可以理解为两个阶段。

### 第一阶段：学习 tokenizer
训练 encoder 和 decoder：

$ x\xrightarrow{E}z\xrightarrow{D}\hat{x}. $

### 第二阶段：学习 latent generative model
在真实图像编码得到的 latent 上训练 diffusion model：

$ z_0\rightarrow z_t\rightarrow z_1. $

生成时：

1. 从噪声开始生成 latent；
2. 得到生成的 latent $ \tilde z $；
3. 使用 decoder 得到图像 $ \tilde x $。

因此，latent diffusion 的最终质量依赖于三个部分：

$ Q=Q(E,D,G). $

其中：

+ $ E $ 决定 latent 保留什么信息；
+ $ G $ 决定 latent 是否容易被生成；
+ $ D $ 决定如何从 latent 恢复图像。

---

## 2.3.3 现有 RAE 和 REPA 方法
论文提到，现代 latent diffusion 方法包括：

+ Rombach 等人的 latent diffusion；
+ RAE；
+ REPA；
+ 其他使用 DINO 或类似 representation encoder 的生成方法。

RAE 和 REPA 使用预训练 representation encoder，例如 DINOv2，再训练 decoder 或 diffusion model。

这些方法取得了很好的生成结果，但论文指出存在一个实验归因问题：

> 如果 encoder 是从一个更大模型和更大数据集预训练得到的，那么最终生成质量提升究竟来自 tokenizer encoder，还是来自后续 decoder 和 generative model，并不容易区分。
>

论文特别指出，DINOv2-B 可能来自一个约 11 亿参数的预训练模型，并且使用了远大于 ImageNet 100 倍的数据量。

如果只报告最终 FID，而不把 encoder 的预训练 FLOPs 计算进去，那么无法公平判断 latent tokenizer 是否具有计算优势。

## 2.3.4 论文对已有 latent diffusion 工作的主要质疑
论文并不是否定 latent diffusion，而是强调三个需要控制的问题。

### 问题一：没有控制 representation learner 的预训练成本
很多工作报告 decoder 和 latent diffusion 的成本，但没有充分计算 DINOv2 等 encoder 的预训练成本。

因此，整体 FLOP efficiency 可能被高估。

### 问题二：encoder 可能不是为生成设计的
DINOv2 等 encoder 主要为了语义和判别任务训练。

它们可以提供很好的 semantic prior，但不保证 patch-level latent 保留所有生成变量。

### 问题三：encoder 的 scaling 可能被 decoder 掩盖
如果 decoder 或后续 diffusion model 规模很大，整体 FID 变好并不代表 encoder 的 latent 更容易建模。

所以需要固定后续生成器，单独测试 encoder 的 diffusability。

---

## 2.3.5 JiT 与 Latent Forcing
论文使用 JiT 作为代表性的 pixel-space diffusion model。

同时使用 Latent Forcing 来评估 representation 的生成价值。Latent Forcing 的基本思想是：

1. 先在 representation 或 latent 空间进行去噪；
2. 再根据去噪后的 latent 直接进行像素生成；
3. 尽量避免额外的有损 decoder 影响；
4. 通过固定后续生成模型来隔离 encoder 的贡献。

这样，作者可以比较：

+ 直接在像素空间训练生成模型；
+ 先学习 representation，再在 latent 上生成。

总计算量可以写成：

$ C_{\mathrm{total}}
=
C_{\mathrm{encoder}}
+
C_{\mathrm{generative\ model}}
+
C_{\mathrm{decoder}}. $

如果 representation learning 真的有用，那么在固定 $ C_{\mathrm{total}} $ 的条件下，加入 tokenizer 应该可以取得更好的生成质量。

## 2.3.6 RePR 在 Pixel/Latent Generative Modeling 中的位置
RePR 试图解决 latent diffusion 中的一个核心问题：

> latent tokenizer 的 encoder 不应该只是一个语义特征提取器，而应该产生容易被生成模型建模的 latent。
>

传统 latent diffusion 主要关注：

+ decoder 是否能恢复图像；
+ latent diffusion 是否能生成 latent；
+ 最终图像 FID 是否良好。

RePR 进一步关注：

+ encoder 是否保留所有生成相关变量；
+ patch features 是否受到直接监督；
+ latent 是否具有良好的 diffusability；
+ encoder 的训练 FLOPs 是否能够带来可预测收益。

因此，论文将 representation learning 视为生成系统中的一个额外 scaling axis：

$ \text{Better representation}
\Rightarrow
\text{Easier latent generation}
\Rightarrow
\text{Better total FLOP efficiency}. $

---

# 2.4 Related Work 三个小节之间的逻辑关系
这三个小节可以按照下面的逻辑理解。

## 第一层：Representation Learning
已有方法如何学习 representation？

+ Contrastive learning 依赖 positive/negative samples；
+ Pixel prediction 预测缺失像素；
+ Latent prediction 预测 teacher representation。

问题是：这些目标可能不完全适合作为生成 tokenizer。

## 第二层：PMI and Representations
为什么可以用统计信息来理解 representation？

+ Word2Vec 与 PMI factorization 有关；
+ Contrastive learning 也可以近似 PMI factorization；
+ RePR 直接使用生成模型估计 PMI，而不是依赖 negative samples 间接估计。

## 第三层：Pixel and Latent Space Generative Modeling
这些 representation 最终如何进入生成模型？

+ pixel diffusion 直接生成图像；
+ latent diffusion 先生成 latent，再通过 decoder 得到图像；
+ tokenizer encoder 的质量和训练成本会影响整个生成系统；
+ RePR 试图让 encoder 本身呈现生成导向的 scaling law。

---

# 2.5 Related Work 的核心结论
Related Work 章节可以总结为：

1. 传统 contrastive learning 擅长学习判别性语义表示，但可能受到 hard negatives 和 feature suppression 的影响。
2. Pixel prediction 不需要 negative samples，但可能被高频细节、像素多解和 likelihood 计算成本影响。
3. Latent prediction 的目标更抽象，但可能继承 teacher representation 的信息选择和 feature suppression。
4. PMI 为对比学习、词向量和 representation learning 提供了统一的统计解释。
5. RePR 将 PMI factorization 与生成模型结合，直接使用条件和无条件生成概率的差异监督 representation。
6. Pixel diffusion 与 latent diffusion 是现代生成建模的两条主要路线。
7. 过去 latent tokenizer 的计算效率和 encoder scaling 还没有被完全隔离和分析。
8. RePR 的定位是：使用生成模型估计 PMI，并把生成任务需要的信息直接写入 patch-level latent representation。

最重要的区分是：

> 传统 representation learning 主要问“哪些特征足以完成判别任务？”
>
> RePR 试图问“一个 view 对预测另一个 view 提供了多少生成相关信息？”
>

对于分类任务，这两个问题可能足够接近；对于生成 tokenizer，它们可能有很大差异。

