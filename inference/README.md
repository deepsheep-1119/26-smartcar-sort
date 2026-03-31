# 推理脚本

存放模型推理相关的脚本文件。

## 目录内容

| 文件 | 说明 |
|------|------|
| `detect_red.py` | 通过红色检测A4纸区域，进行图像透视校正 |
| `predict.py` | MNIST手写数字识别推理脚本 |
| `smartcar_predict.py` | 智能小车目标识别推理脚本 |

## 使用方法

### MNIST 数字识别
```bash
python inference/predict.py <图像路径>
```

### 智能小车目标识别
```bash
python inference/smartcar_predict.py
```

### A4纸红色区域检测
```bash
python inference/detect_red.py
```
