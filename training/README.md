# 训练脚本

存放模型训练相关的脚本文件。

## 目录内容

| 文件 | 说明 |
|------|------|
| `train.py` | MNIST手写数字识别训练脚本 |
| `smartcar_train.py` | 智能小车目标识别训练脚本 |

## 使用方法

### MNIST 训练
```bash
python training/train.py
```
训练完成后模型保存为 `models/mnist_model.pth`

### 智能小车训练
```bash
python training/smartcar_train.py
```
训练完成后模型保存为 `models/smartcar_model.pth`
