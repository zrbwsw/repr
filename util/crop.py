# 中文导读：ADM 风格的等比例缩放与中心裁剪，供训练/参考图像预处理复用。
# 输入和输出都是 PIL 图像；函数不负责 RGB 转换、张量化或数值归一化。
import numpy as np
from PIL import Image


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    # 先反复缩小一半，将很大的图像逐步降到接近目标尺度，采用 BOX 重采样。
    # 循环条件基于短边，整个过程保持原有宽高比例。
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    # 最后用 BICUBIC 将短边精确缩放到 image_size，长边一般仍大于目标尺寸。
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # 在缩放后的数组上，以中心为基准取 image_size×image_size 的窗口。
    # 奇数像素差通过整数除法向下取整，左右/上下最多相差一个像素。
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
