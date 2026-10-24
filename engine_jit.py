import math
import sys

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def _normalize_images(x, device):
    x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
    return x * 2.0 - 1.0


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels, filenames) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = _normalize_images(x, device)
        labels = None if args.dataset == "gpic" else labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, results_dict = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**results_dict)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        reduced_results = {}
        for k, v in sorted(results_dict.items()):
            reduced_results[k] = misc.all_reduce_mean(v)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                for k, v in reduced_results.items():
                    log_writer.add_scalar(k, v, epoch_1000x)
