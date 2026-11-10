"""Stream GPIC image/JSON pairs from Hugging Face without caching shards."""
# 中文导读：GPIC 的远程流式 WebDataset 输入管线；图像与 JSON 元数据成对读取。
# HF_TOKEN 由环境提供，下载命令中保留变量引用，本代码不把令牌值写入文件。
# 该数据路径需要外部数据访问权限；下面的 shard 数量是当前源码的配置常量。
import math
import os
import random

from PIL import Image
import webdataset as wds


REPO = "stanford-vision-lab/gpic"
REVISION = "main"
SHARD_COUNT = 8000


# 随机裁剪比例按边长定义，不是 torchvision RandomResizedCrop 的面积比例。
# 先随机选择缩放后的短边长度，再保持宽高比缩放，最后随机选正方形窗口。
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """GPIC crop fractions apply to side length; preserve aspect ratio when resizing."""
    pil_image = pil_image.convert("RGB")
    # image_size/短边长度近似落在 min_crop_frac..max_crop_frac 内。
    # randrange 上界加 1，使整数化后的最大短边长度也有机会被采到。
    smaller_dim_size = random.randrange(
        math.ceil(image_size / max_crop_frac), math.ceil(image_size / min_crop_frac) + 1
    )
    while min(pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = smaller_dim_size / min(pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    # 两个偏移分别从合法范围均匀抽样，保证裁剪窗口完整落在图像内。
    crop_y = random.randrange(pil_image.height - image_size + 1)
    crop_x = random.randrange(pil_image.width - image_size + 1)
    return pil_image.crop((crop_x, crop_y, crop_x + image_size, crop_y + image_size))


# 返回已经分片、解码、转换并组好 batch 的可重复流，而非固定长度数组。
# 节点/worker 分片避免同一次遍历时各进程读取同一份 shard；epoch 长度由上层控制。
def build_gpic_dataset(args, transform):
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("Export an HF_TOKEN authorized to access GPIC")
    # pipe: 让 WebDataset 从 curl 的标准输出读取 tar，无需先完整缓存 shard。
    # curl 的 -q 禁用默认配置文件，-f 使 HTTP 失败返回错误；handler 决定如何跳过异常。
    paths = [
        'pipe:curl -q --http1.1 -fLsS --connect-timeout 600 '
        '-H "Authorization: Bearer $HF_TOKEN" '
        f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/train/gpic_train_{i:05d}.tar'
        for i in range(SHARD_COUNT)
    ]
    # 可选 gpic_seed 控制全局 shard 列表的预先打乱；未提供时使用 args.seed
    # 作为 WebDataset 的随机种子。两处打乱分别作用于 shard 层和样本缓冲层。
    gpic_seed = getattr(args, "gpic_seed", None)
    if gpic_seed is not None:
        # Shuffle all shards before rank/worker splitting, not just each worker's initial buffer.
        random.Random(gpic_seed).shuffle(paths)
    return (
        wds.WebDataset(paths, shardshuffle=len(paths), detshuffle=True,
                       seed=args.seed if gpic_seed is None else gpic_seed,
                       nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker,
                       cache_size=0, handler=wds.warn_and_continue)
        # 重复流可以持续供给样本；筛选使用 seed 与稳定 sample key 构造独立 RNG，
        # 所以同一个样本在重复访问时具有一致的子集成员关系。筛选概率决定的是期望
        # 子集规模，不保证最终恰好等于 train_subset_size 张。
        .repeat()
        .select(lambda sample: args.train_subset_size is None or random.Random(f"{args.seed}:{sample['__key__']}").random() < min(1.0, args.train_subset_size / args.dataset_size))
        # 1000 是样本级 shuffle 缓冲大小；其后解码为 PIL 图像，并保留 JSON 与 key。
        # map_tuple 只对第一个图像分量应用 transform，其余元数据保持原样。
        .shuffle(1000)
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("jpg;png;jpeg;webp", "json", "__key__", handler=wds.warn_and_continue)
        .map_tuple(transform, handler=wds.warn_and_continue)
        # 上层将 global_batch_repeat 并入模型内的 mask/view 重复轴，复用同一张图像；
        # 因此此处先减小实际加载的图像数量，不必从数据管线重复读取同一图像。
        # partial=False 丢弃不完整 batch，维持后续编译/固定形状操作所需的 batch 大小。
        .batched(args.batch_size // args.global_batch_repeat, partial=False)
    )
