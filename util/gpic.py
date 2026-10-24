"""Stream GPIC image/JSON pairs from Hugging Face without caching shards."""
import math
import os
import random

from PIL import Image
import webdataset as wds


REPO = "stanford-vision-lab/gpic"
REVISION = "main"
SHARD_COUNT = 8000


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """GPIC crop fractions apply to side length; preserve aspect ratio when resizing."""
    pil_image = pil_image.convert("RGB")
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
    crop_y = random.randrange(pil_image.height - image_size + 1)
    crop_x = random.randrange(pil_image.width - image_size + 1)
    return pil_image.crop((crop_x, crop_y, crop_x + image_size, crop_y + image_size))


def build_gpic_dataset(args, transform):
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("Export an HF_TOKEN authorized to access GPIC")
    paths = [
        'pipe:curl -q --http1.1 -fLsS --connect-timeout 600 '
        '-H "Authorization: Bearer $HF_TOKEN" '
        f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/train/gpic_train_{i:05d}.tar'
        for i in range(SHARD_COUNT)
    ]
    gpic_seed = getattr(args, "gpic_seed", None)
    if gpic_seed is not None:
        # Shuffle all shards before rank/worker splitting, not just each worker's initial buffer.
        random.Random(gpic_seed).shuffle(paths)
    return (
        wds.WebDataset(paths, shardshuffle=len(paths), detshuffle=True,
                       seed=args.seed if gpic_seed is None else gpic_seed,
                       nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker,
                       cache_size=0, handler=wds.warn_and_continue)
        .repeat()
        .select(lambda sample: args.train_subset_size is None or random.Random(f"{args.seed}:{sample['__key__']}").random() < min(1.0, args.train_subset_size / args.dataset_size))
        .shuffle(1000)
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("jpg;png;jpeg;webp", "json", "__key__", handler=wds.warn_and_continue)
        .map_tuple(transform, handler=wds.warn_and_continue)
        .batched(args.batch_size // args.global_batch_repeat, partial=False)
    )
