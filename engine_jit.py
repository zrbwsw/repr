import math
import sys

import torch

import util.misc as misc
import util.lr_sched as lr_sched


# 中文导读：本文件处理“一批数据的一次优化”，不实现 mask 或 PMI 公式。
# 每四步如何选择 RePR+MAR/MAR-only，由 denoiser_repr.ReprDenoiser.forward 决定；
# 本循环对任一分支返回的总 loss 都执行同样的一次 backward 和 AdamW step。
def _normalize_images(x, device):
    # 上游 PILToTensor 保留 RGB 像素的 0..255 量纲，先搬到设备并转为 FP32 再缩放。
    # non_blocking 配合 DataLoader 的 pinned memory 可以异步传输；这里不做 ImageNet
    # mean/std 标准化。若调用者提供的输入已经在 [0,1]，直接复用此函数会再次错误缩放。
    x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
    return x * 2.0 - 1.0


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    # model 是 DDP wrapper，model_without_ddp 是其 module，后者用于访问自定义方法。
    # train(True) 递归设置子模块；第二轮的 frozen teacher 会在实际取特征前单独重设 eval。
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    # 清除进入 epoch 时残留的梯度；循环内每次 backward 前还会再次清零。
    # 这里没有按若干 microbatch 累积梯度的机制。
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # WebDataset 产出图像、类别/元数据、样本 key。filenames 在该训练函数中不参与损失。
    # data_loader 的长度来自 main_jit.py 的虚拟 epoch 步数，不一定等于一遍真实数据集。
    for data_iter_step, (x, labels, filenames) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        # 用 epoch + 当前步/epoch步数作为连续时间，逐步 warmup/衰减而不是整 epoch 跳变。
        # adjust_learning_rate 还会把基础 LR 乘上参数组的 μP lr_scale，必须保留这些分组字段。
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = _normalize_images(x, device)
        # GPIC 的第二项是 JSON 元数据，不是分类 tensor，因此直接用 None。
        # ImageNet 标签仍搬到 GPU 以兼容接口，但当前 ReprDenoiser 的自监督分支不使用标签。
        labels = None if args.dataset == "gpic" else labels.to(device, non_blocking=True)

        # 仅 forward 在 CUDA BF16 autocast 上下文中运行；模型的 @torch.compile 在其内部生效。
        # autocast 不等于把所有参数永久变成 BF16，具体算子由混合精度规则选 dtype；
        # 主入口 model.to(device) 未显式降参数精度，本循环也没有 GradScaler。
        # 此处设备类型写死为 'cuda'，后面还有 cuda.synchronize，不能仅改 --device 就当成 CPU 训练。
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, results_dict = model(x, labels)

        # loss 保持可微 tensor 供反传，loss_value 是用于有限性检查/日志的 Python 数。
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 每批一次清梯度 -> 反传 -> 更新；repeat 产生的多个视图已在模型内部合成总 loss。
        # backward 通过 DDP 完成梯度同步。这里没有额外 gradient clipping 或 loss scaling，
        # 也没有用 retain_graph 再反传某一个子损失。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 等待当前 CUDA 工作完成后再记录后续时间/指标；这里显式让 CPU 等待 GPU 完成。
        torch.cuda.synchronize()

        # 当前 ReprDenoiser.update_ema() 只有 pass；调用此接口不会产生 EMA 权重，
        # 也不会更新第二轮的 frozen teacher。
        model_without_ddp.update_ema()

        # MetricLogger 这里接收的是本 rank 的值，用于平滑窗口/控制台展示。
        # results_dict 随任务变化：联合步有空间/全局表征损失，MAR-only 步没有这些子项。
        metric_logger.update(loss=loss_value)
        metric_logger.update(**results_dict)
        # 只记录第一个参数组的 LR；μP 不同参数组可能具有不同实际 LR。
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # 另做跨 rank 的标量均值供 TensorBoard 使用；这不是 backward 的梯度 all-reduce。
        # 所有 rank 都执行归约，即使只有 rank 0 持有 log_writer；任务步调应保持一致。
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        reduced_results = {}
        for k, v in sorted(results_dict.items()):
            # 按 key 排序，使各 rank 对子指标调用 collective 的顺序一致。
            reduced_results[k] = misc.all_reduce_mean(v)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            # 横坐标为“虚拟 epoch*1000”的整数，不是优化步计数；log_freq 用本 epoch 的步号取模。
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                for k, v in reduced_results.items():
                    log_writer.add_scalar(k, v, epoch_1000x)
    # 沿用上游行为：本函数没有返回 epoch 汇总字典，也没有在结尾额外同步 MetricLogger。
