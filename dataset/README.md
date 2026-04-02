# 数据集脚本

存放数据集处理相关的脚本文件。

## 目录内容

| 文件 | 说明 |
|------|------|
| `scripts/organize_dataset.py` | 数据集整理脚本，将数据集划分为train/test |

## 使用方法

### 数据集整理

```bash
python dataset/scripts/organize_dataset.py <源目录> --output <输出目录> --ratio <训练集比例>
```

## 数据集结构

训练脚本使用 `data/smartcar/train/` 和 `data/smartcar/test/` 目录下的数据：

```
data/smartcar/
├── train/
│   ├── 交通工具-直行/
│   ├── 武器-左/
│   └── 物资-右/
└── test/
    ├── 交通工具-直行/
    ├── 武器-左/
    └── 物资-右/
```

原始图像数据存放在 `png_smartcar/` 目录下，按类别分类。
