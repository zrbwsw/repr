# 中文导读：训练日志、分布式环境、参数分组和 checkpoint 的通用辅助函数。
# 部分函数沿用旧 JiT 训练方式，尤其 save_model 依赖双 EMA 列表；
# RePR 主入口使用自己的保存字典配合 save_on_master，不调用该旧保存器。
import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import copy

import torch
import torch.distributed as dist


# 同时维护最近 window_size 次更新的滑动窗口，以及自创建以来的加权总和。
# 窗口 median/avg 用于观察近期波动；global_avg 使用 total/count。
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    # n 是本次值代表的样本/计数权重，只影响全局 total/count。
    # 滑动窗口只 append 一次 value，因此窗口平均与加权全局平均可能不同。
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        # 跨进程只归约 count 和 total；每个进程自己的窗口 deque 保持不变。
        # 使用 CUDA 张量适配下方 NCCL 通信，因此这一路依赖 CUDA。
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


# 按指标名自动创建 SmoothedValue，提供批量更新、字符串展示及遍历进度日志。
# 只接受标量数值；传入 Tensor 时通过 item() 读取单元素值。
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    # 允许 logger.loss 等属性式访问，但底层真实存储仍在 meters 字典中。
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    # 生成器包装器：yield 把当前 batch 交给训练循环，返回后再统计该次迭代耗时。
    # 依赖 len(iterable) 计算进度和 ETA；无限数据流需要上层提供有长度的 epoch 包装。
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            # yield 前的时间主要是取数等待；yield 返回后的总间隔包含本轮训练计算。
            # CUDA 显存日志使用 max_memory_allocated，表示累计峰值而非当前占用。
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# 替换全局 print，添加时间戳，并控制非主进程输出。
# 源码保留一个例外：world_size>8 时会强制允许所有进程打印，见 force 条件。
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


# 同时检查分布式包可用性与进程组是否已经初始化，供单卡/多卡共用逻辑。
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# 未初始化分布式时视为单进程，返回 1；不依据可见 GPU 的数量推断。
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


# 未初始化分布式时返回 rank 0，使单进程也能走主进程保存和日志逻辑。
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# 本项目把全局 rank 0 定义为主进程，不是每个节点的 local rank 0。
def is_main_process():
    return get_rank() == 0


# 只有全局主进程执行 torch.save，避免 DDP 的多个进程同时写同一路径。
def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


# 依次识别 ITP/OpenMPI、torchrun 环境变量、SLURM 三种启动方式。
# 普通单进程会设置 distributed=False 并返回；多进程设置当前 CUDA 设备、
# 创建 NCCL 进程组并 barrier 同步。此函数不负责自行启动多个进程。
def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    # torchrun 会注入全局 RANK/WORLD_SIZE 和本节点 LOCAL_RANK。
    # gpu 使用 LOCAL_RANK，避免把跨节点的全局 rank 误当成本机设备编号。
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # 该分支从 SLURM_PROCID 推导 rank 和本机 GPU；world_size/dist_url 仍依赖 args。
    # 这里保持原逻辑，不额外猜测或修补集群启动参数。
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# 通用旧参数分组：冻结参数跳过，1D 参数、bias、skip_list 与名称含 diffloss
# 的参数不做 weight decay，其余使用给定衰减。RePR 当前使用模型自己的 μP
# 分组函数，不能把这里的名称规则误认为 RePR 的最终 optimizer 分组。
def add_weight_decay(model, weight_decay=0, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


# 旧 JiT checkpoint 保存器：保存在线模型、optimizer、epoch/args 和两套 EMA。
# 要求调用者已经建立 ema_params1/2；ReprDenoiser 没有这两套列表，因此主入口
# 另行构造保存内容。这里保留原函数供旧路线使用。
def save_model(args, model_without_ddp, optimizer, epoch, epoch_name=None):
    if epoch_name is None:
        epoch_name = str(epoch)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)

    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }

    # ema
    # 先深拷贝完整 state_dict，再按 named_parameters 的顺序替换 EMA 参数。
    # buffers 仍来自在线模型快照，并没有在此单独计算滑动平均。
    ema_state_dict1 = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict2 = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict1 and name in ema_state_dict2
        ema_state_dict1[name] = model_without_ddp.ema_params1[i]
        ema_state_dict2[name] = model_without_ddp.ema_params2[i]
    to_save['model_ema1'] = ema_state_dict1
    to_save['model_ema2'] = ema_state_dict2

    save_on_master(to_save, checkpoint_path)


# 把各 rank 的标量做等权平均，返回 Python 数值，常用于日志。
# 它不会按每个 rank 的样本量加权，也不是用于反向传播的可微通信操作。
def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x