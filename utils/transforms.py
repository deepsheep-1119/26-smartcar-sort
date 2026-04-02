from torchvision import transforms


def get_smartcar_transform(img_size=96, train=True):
    """
    获取小车门控数据集的图像预处理流水线。

    Args:
        img_size: 目标图像尺寸，默认为 96x96
        train: 是否为训练模式。默认为 True。

    Returns:
        transforms.Compose: 图像预处理流水线
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                # 1. 随机旋转：在正负5度之间随机旋转图像，帮助模型学习旋转不变性
                transforms.RandomRotation(degrees=5),
                # 2. 随机调整亮度、对比度、饱和度：
                #    - brightness=0.2 表示亮度在 [0.8, 1.2] 之间随机变动
                #    - contrast=0.2 表示对比度在 [0.8, 1.2] 之间随机变动
                #    - saturation=0.1 表示饱和度在 [0.9, 1.1] 之间随机变动
                #    模拟不同光照条件下的场景
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                # 3. 随机水平翻转：50%概率翻转，帮助模型学习左右对称性
                transforms.RandomHorizontalFlip(p=0.5),
                # 转换为 Tensor：归一化到 [0, 1] 范围
                transforms.ToTensor(),
                # 正则化：将像素值映射到 [-1, 1] 范围，有助于训练稳定性
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        # 测试/验证模式：不做任何随机变换，保证结果可重复
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )


def get_smartcar_predict_transform(img_size=96):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
