# 数据集脚本

存放数据集处理和图片分类相关的脚本文件。

## 目录内容

| 文件 | 说明 |
|------|------|
| `organize_dataset.py` | 通用数据集整理脚本，将数据集划分为train/test |
| `split_dataset.py` | 智能小车数据集专用划分脚本 |
| `prepare_dataset.py` | 数据集准备脚本，从out目录复制到data/smartcar |
| `smartcar_dataset.py` | 智能小车数据加载器 |

## 使用方法

### 通用数据集整理
```bash
python dataset/organize_dataset.py <源目录> --output <输出目录> --ratio <训练集比例>
```

### 智能小车数据集划分
```bash
python dataset/split_dataset.py
```

### 准备智能小车训练数据
```bash
python dataset/prepare_dataset.py
```

## 数据集结构

处理后的数据集存放在 `data/smartcar/` 目录下：
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
