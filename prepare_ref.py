# 中文导读：把 ImageFolder 格式的 ImageNet train 集预处理并导出 PNG。
# 这是参考图像准备脚本，不执行 RePR 训练，也不直接计算 FID 统计量。
# 输入目录需要是 data_path/train/类别目录/图像，输出是按全局编号排列的平坦目录。
import os
import argparse
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from util.crop import center_crop_arr


# 解析路径与分辨率，建立中心裁剪的数据管线，按 batch 读取后逐张保存。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNet root directory')
    parser.add_argument('--output_path', type=str, default='imagenet-train-256',
                        help='Folder where transformed images will be saved')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Resolution to center-crop and resize')
    args = parser.parse_args()

    # 先调用 ADM 风格中心裁剪，再用 CenterCrop 保持目标尺寸。
    # ToTensor 给出 [0,1] 浮点值；这里没有训练用的 [-1,1] 归一化或随机增强。
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, 'train'),
        transform=transform_train
    )

    # 按 ImageFolder 的确定顺序遍历，shuffle=False；标签参与加载但不写入输出文件。
    # batch_size=256、num_workers=32 是脚本中的固定参数，非命令行可调项。
    data_loader = DataLoader(
        dataset_train,
        batch_size=256,
        num_workers=32,
        shuffle=False,
        pin_memory=False
    )

    os.makedirs(args.output_path, exist_ok=True)

    to_pil = transforms.ToPILImage()
    # 全局计数跨 batch 累加，因此同一轮导出的文件名不会在 batch 边界重置。
    # 若重用同一输出目录，已有相同名称的 PNG 会被 Pillow 覆盖。
    global_idx = 0

    from tqdm import tqdm
    for batch_images, batch_labels in tqdm(data_loader):
        for i in range(batch_images.size(0)):
            img_tensor = batch_images[i]

            pil_img = to_pil(img_tensor)
            out_path = os.path.join(
                args.output_path,
                f"transformed_{global_idx:08d}.png"
            )
            # PNG 为无损格式；compress_level=0 选择不做压缩，节省压缩计算但文件更大。
            pil_img.save(out_path, format='PNG', compress_level=0)
            global_idx += 1

        print(f"Saved batch up to index={global_idx} ...")

    print("Finished saving all images.")


if __name__ == "__main__":
    main()