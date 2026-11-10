# 中文导读：这是仓库保留的原始 JiT 图像去噪/生成包装器。
# RePR 当前训练入口 main_jit.py 使用 denoiser_repr.ReprDenoiser；本文件的
# 类别条件、CFG、ODE 采样和双 EMA 不应直接当成 Figure 1 的 RePR 实现。
# 本文件调用的是 model_jit.JiT，输出整张图像的干净预测 x_pred。
import torch
import torch.nn as nn
from model_jit import JiT_models


# 把 JiT 主干、训练噪声分布、类别丢弃、采样器和 EMA 更新封装在一起。
# args 提供配置；模型构造时同时确定图像尺寸和类别 embedding 表大小。
class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        # 这些配置决定训练时的随机变量：label_drop_prob 是逐样本类别丢弃概率，
        # P_mean/P_std 是 logit(t) 的正态分布参数，noise_scale 是高斯噪声幅度。
        # t_eps 用于限制 1/(1-t) 的放大程度，不改变 t 的采样范围。
        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        # 此处只放占位符；调用 update_ema 前，外部训练代码必须准备对应参数列表。
        # 这与 ReprDenoiser.update_ema 的空实现不同。
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    # 对 batch 中每个标签独立抽样；被丢弃的标签改成 num_classes。
    # 正常类别索引为 0..num_classes-1，额外的最后一行 embedding 表示空条件。
    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    # 先采样正态变量，再取 sigmoid，得到 (0,1) 内的 logit-normal 时间。
    # 这里 t 是干净图像的权重：t 接近 0 时噪声强，接近 1 时图像干净。
    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    # 训练输入 x 为 [B,C,H,W]，labels 为 [B]；返回对像素和 batch 平均的标量损失。
    # 仅训练模式执行类别 dropout，网络始终直接预测干净图像，损失则按速度形式计算。
    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        # 把每张图一个时间扩展成 [B,1,1,1]，使它广播到所有通道/像素。
        # e 的标准差由 noise_scale 控制；它与干净图像按 t 线性混合。
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        # 插值路径 z_t=t*x+(1-t)*e；当分母未被截断时，真实速度等于 x-e。
        # 预测速度与真实速度之差等于 (x_pred-x)/max(1-t,t_eps)，因此下方的
        # 速度 MSE 也可理解成带时间权重的干净图像预测误差。
        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    # 从 t=0 的噪声沿 ODE 积分到 t=1，生成给定类别的图像。
    # 这是 JiT 生成路径；RePR 的 forward_feats 提取表征并不调用此采样器。
    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        # 时间网格有 steps+1 个端点，形状为 [steps+1,B,1,1,1]。
        # 每个积分区间的长度是 t_next-t，网络前向时再将当前时间展平成 [B]。
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # 最后一个区间统一使用 Euler，避免 Heun 校正时在终点 t=1 再估计一次速度。
        # 即使选择 heun，前 steps-1 个区间与最后一个区间的求解方式仍不同。
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    # 同一个 z,t 分别做有类别和空类别两次前向，再在速度空间实施 CFG。
    # 区间外把 guidance 系数设成 1，返回有类别预测；并不是返回空类别预测。
    # 当前实现无论区间是否启用 CFG，都会计算这两个分支。
    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        # 边界按源码严格处理：上界 high 不包含；low>0 时下界也不包含。
        # 只有 low==0 的特殊情况允许 t=0 落入启用区间，读采样步数时应据此判断。
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    # Euler：用区间起点的一次速度估计，向前推进 dt。
    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    # Heun：先用 Euler 预测终点，再计算终点速度，取两端速度的平均进行校正。
    # 每次 _forward_sample 又包含有条件/空条件两次网络前向。
    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    # 分别以两个衰减率更新参数滑动平均：ema <- decay*ema + (1-decay)*online。
    # 只遍历 parameters；本函数不负责建立列表、切换采样权重或更新 buffers。
    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
