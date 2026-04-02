# 训练脚本

存放模型训练相关的脚本文件。

## 目录内容

| 文件 | 说明 |
|------|------|
| `smartcar_train.py` | 智能小车目标识别训练脚本 |

## 使用方法

```bash
python training/smartcar_train.py
```

训练完成后模型保存为 `smartcar_model.pth`（项目根目录）

## 模型架构

3层CNN + FC:
- Conv2d(1, 32) -> ReLU -> MaxPool
- Conv2d(32, 64) -> ReLU -> MaxPool
- Conv2d(64, 128) -> ReLU -> MaxPool
- FC(128 * 12 * 12, 256) -> ReLU -> Dropout(0.5)
- FC(256, 3)
