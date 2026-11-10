import torch
import torch.nn as nn
from model_repr import REPR_MODELS


# 中文导读：本文件是训练任务的包装层，不实现 E/D/表征头/扩散 MLP 的内部计算。
# main_jit.py 负责数据、DDP、优化器与 checkpoint；这里决定本步调用哪个模型分支，
# model_repr.py 的 REPR 则实现真实网络、mask、生成损失与 PMI 表征损失。
class ReprDenoiser(nn.Module):
    def __init__(self, args):
        super().__init__()

        # REPR_MODELS 的工厂函数决定主干宽度、头数与层数；args 是 main_jit.py 的 CLI 配置。
        # 不要把 REPR 类自身默认值、CLI 默认值、README 显式训练参数当成同一组配置。
        self.net = REPR_MODELS[args.model](
            input_size=args.img_size,
            in_channels=3,
            # in_context_len 是可学习 register/in-context token 数，不是文本条件长度。
            # repr_dim 是最终每个空间/register 表征的维数，不等于主干 hidden_size。
            in_context_len=args.in_context_len,
            repr_dim=args.repr_dim,
            diff_mlp_depth=args.diff_mlp_depth,
            # 主训练入口每卡只读 batch_size/global_batch_repeat 张图；在模型内部增加
            # 同一图像的不同 mask 视图。这里的 repeat 是视图维度，不是多次 optimizer.step。
            # 例如 README 第一轮每卡 batch_size=256、global_batch_repeat=8、batch_repeat=1，
            # 实际传入 32 张图，模型内部构造每图 8 组 A/AB 视图，共 256 组。
            batch_repeat=args.batch_repeat * args.global_batch_repeat,
            diff_repeat=args.diff_repeat,
            p_repeat=args.p_repeat,
            diff_repeat_repr=args.diff_repeat_repr,
            b_max_visible=args.b_max_visible,
            a_min_mask=args.a_min_mask,
            mask_strategy_mar=args.mask_strategy_mar,
            mask_strategy_repr=args.mask_strategy_repr,
            repr_loss_weight=args.repr_loss_weight,
            repr_delta=args.repr_delta,
            pad_dropout=args.pad_dropout,
            mu_repr=args.mu_repr,
            std_repr=args.std_repr,
            mu_mar=args.mu_mar,
            std_mar=args.std_mar,
            # MAR-only 分支有自己的重复次数；它与联合分支的 batch_repeat 分开配置。
            batch_repeat_mar=args.batch_repeat_mar * args.global_batch_repeat,
            is_first_iter=args.is_first_iter,
        )

        # 第二轮（或后续轮）才额外构造固定 teacher。student 的输入仍是 RGB 图像，
        # 变化的是生成目标：从第一轮的像素 patch 改为 teacher 的空间表征。
        if args.denoiser_mode == "repr_mar_featspred":
            self.frozen_net = REPR_MODELS[args.frozen_model](
                input_size=args.img_size,
                in_channels=3,
                # 此处按实现固定为 16，不读取 student 的 args.in_context_len；teacher
                # checkpoint 的结构必须与这些构造参数匹配，权重稍后在 main_jit.py 加载。
                in_context_len=16,
                repr_dim=args.repr_dim,
                diff_mlp_depth=args.diff_mlp_depth,
                batch_repeat=args.batch_repeat,
                diff_repeat=args.diff_repeat,
                p_repeat=args.p_repeat,
                diff_repeat_repr=args.diff_repeat_repr,
                b_max_visible=args.b_max_visible,
                a_min_mask=args.a_min_mask,
                mask_strategy_mar=args.mask_strategy_mar,
                mask_strategy_repr=args.mask_strategy_repr,
                repr_loss_weight=args.repr_loss_weight,
                repr_delta=args.repr_delta,
                pad_dropout=args.pad_dropout,
                mu_repr=args.mu_repr,
                std_repr=args.std_repr,
                mu_mar=args.mu_mar,
                std_mar=args.std_mar,
                batch_repeat_mar=args.batch_repeat_mar,
                # teacher 自己是第几轮模型，与当前 student 的 is_first_iter 分开设置；
                # 若以第二轮 teacher 训练第三轮，此开关应对应 teacher 的结构。
                is_first_iter=args.frozen_is_first_iter, # Frozen only parameter
            )
            # 固定 checkpoint teacher 不参与反向传播，也不是随 student 更新的 EMA teacher。
            self.frozen_net.requires_grad_(False)

        self.args = args
        self.denoiser_mode = args.denoiser_mode
        # 包装层保存的是未乘 global_batch_repeat 的值，用于下方截取输入 batch。
        self.batch_repeat = args.batch_repeat
        self.batch_repeat_mar = args.batch_repeat_mar
        # 注册为 buffer 后会随 state_dict 保存/恢复，并随 model.to(device) 移动。
        # 计数跨 epoch 连续，不在每个 epoch 重新从 0 开始；仅联合训练模式递增它。
        self.register_buffer("train_step", torch.zeros((), dtype=torch.long))
        # 对 teacher 输出乘常数，改变第二轮预测目标的数值尺度；不做逐通道标准化。
        self.teacher_feats_mul = args.teacher_feats_mul

    def get_optimizer_param_groups(self, weight_decay, lr, effective_batch_size):
        # 只返回 student 的 μP 参数分组，frozen_net 不会进入优化器。
        # lr 与 effective_batch_size 只保留在接口中，本方法实际没有使用这两个参数；
        # 基础 LR 在 main_jit.py 计算，分组 lr_scale 由 util/lr_sched.py 每步应用。
        # get_mup_param_groups 以宽度 768 为基准，MuPLinear 的矩阵权重 LR 乘 768/宽度，
        # 其他参数 LR 倍率为 1；一维参数与名称以 .bias 结尾的参数不施加 weight decay。
        return self.net.get_mup_param_groups(weight_decay)

    def forward_repr(self, x, x_target=None):
        # 此调用本身已包含 MAR 生成损失，以及空间/全局两种 PMI 表征损失。
        # x_target=None 表示像素目标；给定 x_target 时采用 teacher feature 目标。
        loss, mse_loss, repr_loss, repr_loss_global = self.net.forward_repr(x, x_target)
        # 用于 backward 的 loss 保持 tensor；其余指标转 Python 标量只用于记录。
        # 名为 mse_loss 的生成项在底层实际采用大 delta 的 Huber 形式，不能仅凭名称
        # 认为所有误差范围都严格等价于 MSE；表征项同样使用配置的 Huber delta。
        return loss, {
            'mse_loss': mse_loss.item(),
            'repr_loss': repr_loss.item(),
            'repr_loss_global': repr_loss_global.item(),
        }

    def forward_mar(self, x):
        # 独立 --denoiser_mode mar 的简单接口；本分支不会递增 train_step，且不返回子指标。
        return self.net.forward_mar(x), {}

    @torch.no_grad()
    def get_stats(self, x, labels=None):
        # 只提取 student 表征；labels 没有参与计算，D 和扩散 MLP 也不参与 forward_feats。
        # eval() 会改变 self.net 的模式，本函数未恢复原模式；主入口的统计任务随后即退出。
        self.net.eval()
        spatial, global_features = self.net.forward_feats(x)
        # forward_feats 默认包含图像外一圈 query；先按 padded 网格还原，再裁掉外圈。
        # 256 输入、16 patch 的常用配置中，是从 18x18 裁成 16x16，然后把图像/位置合并。
        spatial = spatial.reshape(x.size(0), self.net.pad_hw, self.net.pad_hw, self.net.repr_dim)[:, self.net.factor_pad:self.net.factor_pad+self.net.hw_len, self.net.factor_pad:self.net.factor_pad+self.net.hw_len].reshape(-1, self.net.repr_dim).float()
        # global_features 原来将多个 register 拼接；此处又把 register 作为样本轴展开。
        # 两类统计都输出 repr_dim 个通道的 mean/std，不是每张图一个均值或一个标准差。
        features = {"spatial": spatial, "global": global_features.reshape(-1, self.net.repr_dim).float()}
        # 这里只统计传入的一个 batch，未做跨进程 all-reduce，也不遍历完整训练集。
        return {f"{name}_{stat}": getattr(feats, stat)(0) for name, feats in features.items() for stat in ("mean", "std")}

    def forward_repr_mar(self, x):
        # 先读取旧 step 再加一，所以全新训练第 0、4、8... 步进入联合分支，
        # 其余 3/4 步只训练 MAR。这个比例指优化步数，不表示两类任务耗时或 FLOPs 相同。
        step = int(self.train_step.item())
        self.train_step.add_(1)

        if step % 4 == 0:
            # 截取前 1/batch_repeat 的独立图像，再在 net 内重复 mask。
            # 若每卡传入 b 张图，联合分支的 A 视图数约为
            # floor(b/batch_repeat) * (batch_repeat*global_batch_repeat)。
            # 代码只用整除截取，未在这里检查 batch_repeat 的正值或可整除性。
            x = x[:x.size(0)//self.batch_repeat]
            return self.forward_repr(x)
        else:
            # 第一轮此处不按 batch_repeat_mar 截取 x。因此当 batch_repeat_mar>1 时，
            # MAR 的内部视图数会继续增加；默认 1 和 README 的 1 不受这个差异影响。
            loss = self.net.forward_mar(x)
            return loss, {'mse_loss': loss.item()}

    def forward_repr_mar_featspred(self, x):
        # 第二轮沿用同一个每四步调度，但两个分支都先从固定 teacher 取得目标特征。
        step = int(self.train_step.item())
        self.train_step.add_(1)

        if step % 4 == 0:
            x = x[:x.size(0)//self.batch_repeat]
            with torch.no_grad():
                # 外层每个 epoch 会 model.train(True)，所以这里再次显式将 teacher 置为 eval。
                # return_padded=False 使目标只对应图像内 N 个 patch，与像素 patch 网格对齐。
                self.frozen_net.eval()
                x_target, _ = self.frozen_net.forward_feats(x, return_padded=False)
            # 只使用 teacher 的空间输出；第二个返回值（全局 register 表征）被丢弃。
            return self.forward_repr(x, x_target * self.teacher_feats_mul)
        else:
            # 与上面的第一轮实现不同，第二轮 MAR-only 会先按 batch_repeat_mar 截取，
            # 再由 student 内部 batch_repeat_mar*global_batch_repeat 个 mask 视图展开。
            x = x[:x.size(0)//self.batch_repeat_mar]
            with torch.no_grad():
                self.frozen_net.eval()
                x_target, _ = self.frozen_net.forward_feats(x, return_padded=False)
            loss = self.net.forward_mar(x, x_target * self.teacher_feats_mul)
            return loss, {'mse_loss': loss.item()}

    def set_epoch(self, epoch):
        # 只调整表征分支外圈 query 的 pad_dropout，既不是整网 dropout 也不是 LR。
        # start=None 时直接使用最终值；start>end 时从第 0 个 epoch 的 start 线性下降，
        # 到 decay_epochs 达到 end，之后固定。该值由每个 epoch 的配置重算，不是 tensor buffer。
        end = self.args.pad_dropout
        start = self.args.pad_dropout_start
        if start is None:
            start = end
        decay_epochs = self.args.pad_dropout_decay_epochs
        # 按现有分支顺序，start<=end 会一直采用 start，并不会执行“向上增长”的日程。
        # 当 decay_epochs<=0 且 start>end 时，第 0 个 epoch 仍取 start，后续 epoch 才取 end。
        if start <= end or epoch <= 0:
            pad_dropout = start
        elif decay_epochs <= 0 or epoch >= decay_epochs:
            pad_dropout = end
        else:
            pad_dropout = start + (end - start) * epoch / decay_epochs
        self.net.pad_dropout = pad_dropout

    def forward(self, x, labels):
        # 此实现是无类别条件的自监督学习：为兼容训练接口保留 labels，但各分支不使用它。
        # CLI 中的 get_stats 在 main_jit.py 单独调用 get_stats()，不会经过这里的分支选择。
        if self.denoiser_mode == "mar":
            return self.forward_mar(x)
        elif self.denoiser_mode == "repr_mar":
            return self.forward_repr_mar(x)
        elif self.denoiser_mode == "repr_mar_featspred":
            return self.forward_repr_mar_featspred(x)
        raise ValueError(f"Unknown training mode: {self.denoiser_mode}")

    def update_ema(self):
        # 保留训练引擎所需的接口，但没有执行任何 EMA 权重更新。
        # 第二轮 frozen_net 始终来自指定 checkpoint，不能把此空实现理解为更新 teacher。
        pass
