## 0.1 章节定位
本章对应论文摘要，回答三个问题：论文提出了什么方法、解决了什么问题、取得了什么结果。

论文的核心任务是学习一种能够作为生成模型 tokenizer 的自监督 representation。这里的 tokenizer 不只是服务于分类或检索，而是要把图像转换成适合后续生成模型建模的 latent tokens。

## 0.2 研究问题
现有表示学习方法，例如 DINO、DINOv2、DINOv3 和 LeJEPA，通常针对判别任务设计。它们倾向于学习对分类、实例识别或语义匹配有用的变量。

但是，生成模型需要保留更多信息，包括：

+ 语义和物体类别；
+ 物体形状与空间布局；
+ 背景和场景结构；
+ 颜色、材质和纹理；
+ 局部 patch 信息；
+ 高频细节和其他影响图像分布的潜变量。

因此，一个分类性能很好的 representation 不一定是一个适合作为生成模型 tokenizer 的 representation。

论文还关注 tokenizer 的计算效率问题：如果 encoder 的预训练成本远高于后续生成模型，那么使用 tokenizer 是否真的比直接进行 pixel-space generation 更高效，需要从 encoder 和生成模型的总 FLOPs 角度重新评估。

## 0.3 方法：Relative Prediction Representations
论文提出 Relative Prediction Representations，简称 RePR。

RePR 的基本思路是：

1. 从同一个数据样本构造两个 masked views；
2. 使用生成模型估计一个 view 对另一个 view 的预测增益；
3. 用这个预测增益构造两个 view 之间的 PMI；
4. 直接监督 representation，使 representation 的内积匹配 PMI；
5. 将这种监督施加到 patch-level features 上。

点互信息定义为：

$ \operatorname{PMI}(x,y)
=
\log \frac{p(y\mid x)}{p(y)}
=
\log p(y\mid x)-\log p(y). $

其中：

+ $ p(y\mid x) $ 表示给定 $ x $ 后预测 $ y $ 的概率；
+ $ p(y) $ 表示不使用 $ x $ 时预测 $ y $ 的概率；
+ 两者之差表示 $ x $ 对预测 $ y $ 提供的额外信息。

RePR 的表示目标可以概括为：

$ \langle R_\theta(x),R_\theta(y)\rangle
\approx
\operatorname{PMI}(x,y). $

论文进一步使用加权 PMI、增量式 PMI 和特征空间的迭代预测，使监督信号更加适合高维图像以及生成tokenizer。

## 0.4 论文声称解决的问题
### 1. Feature suppression
判别式目标可能只保留完成判别任务所需的少量信息，而压制生成所需的其他变量。

例如，如果一个 watermark 已经足以区分不同图像，那么对比目标可能只学习 watermark，而不再学习图像真实内容。

### 2. Patch feature 缺少直接监督
DINO 等方法主要监督 CLS token，但生成 tokenizer 通常使用 encoder 内部的 patch features。

这些 patch features 可能随着模型规模扩大出现 outlier、漂移、各向异性和空间结构不稳定。

### 3. 负样本依赖
对比学习依赖 negative samples，在 strongly conditioned views 下容易遇到 hard negative 问题。RePR 通过条件和无条件生成概率之差直接估计 PMI，不需要依赖大量 negative samples。

## 0.5 主要实验结果
论文在 ImageNet 和 GPIC 上进行实验。

主要结果包括：

+ RePR 在 watermark 设置下比 DINOv3 和 LeJEPA 更不容易出现 feature suppression；
+ RePR encoder 的规模增加后，latent diffusability 随训练 FLOPs 增加而改善；
+ 在 ImageNet 和 GPIC 上观察到近似 power-law scaling；
+ 与 Latent Forcing 结合后，在总 FLOPs 相近的条件下优于直接 pixel diffusion；
+ 使用 JiT-L 时，论文报告 FID-50K 为约 $ 2.18 $；
+ 外推结果显示，达到 DINOv2 类似 ImageNet 生成质量所需的 encoder FLOPs 约为 DINOv2 的 $ 1/7 $，达到 WebSSL 类似 GPIC 质量所需 FLOPs 约为其 $ 1/8 $。

这些 $ 1/7 $ 和 $ 1/8 $ 的数字是论文在有限测量区间上的经验外推，不应理解为无限规模下的严格定理。

## 0.6 摘要的核心结论
摘要真正想表达的不是某一个 FID 数字，而是下面这个结论：

> 如果 representation 的训练目标与生成任务需要保留的信息保持一致，那么 tokenizer encoder 本身也可以具有可预测的 scaling law。
>

换句话说，encoder 并不是天然无法 scaling。问题在于，传统 encoder 的训练目标往往是判别式的，模型规模扩大后可能更有效地学习 shortcut 和少量语义 bit，而不是更完整的生成信息。

