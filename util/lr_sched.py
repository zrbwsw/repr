import math


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according to the configured schedule."""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif args.lr_schedule == "wsd":
            decay_start = args.epochs - args.decay_epochs
            if args.decay_epochs == 0 or epoch < decay_start:
                lr = args.lr
            else:
                lr = args.min_lr + (args.lr - args.min_lr) * (args.epochs - epoch) / args.decay_epochs
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
