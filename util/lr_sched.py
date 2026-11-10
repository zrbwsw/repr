# 中文导读：逐训练步计算基础学习率，并保留每个参数组的 lr_scale。
# RePR 的 μP 参数分组会设置 lr_scale；调度器不能把所有组简单覆盖成同一学习率。
import math


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according to the configured schedule."""
    # epoch 可以是带小数的训练进度（当前 epoch + step/steps_per_epoch）。
    # warmup 从 0 线性升至 args.lr；warmup_epochs=0 时条件不会进入此除法分支。
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        # 余弦仅覆盖 warmup 结束后的剩余区间，终点降到 min_lr。
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        # WSD = warmup/stable/decay：warmup 已在前面处理，中间保持 args.lr，
        # 最后 decay_epochs 个 epoch 线性降到 min_lr；decay_epochs=0 表示不衰减。
        elif args.lr_schedule == "wsd":
            decay_start = args.epochs - args.decay_epochs
            if args.decay_epochs == 0 or epoch < decay_start:
                lr = args.lr
            else:
                lr = args.min_lr + (args.lr - args.min_lr) * (args.epochs - epoch) / args.decay_epochs
        else:
            raise NotImplementedError
    # 返回的是未乘 lr_scale 的基础 lr；每组真正使用的是 lr*lr_scale。
    # 这样保持模型宽度相关的 μP 学习率比例随整个调度一起变化。
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
