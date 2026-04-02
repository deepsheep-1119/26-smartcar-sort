# 推理脚本

存放模型推理相关的脚本文件。

## 目录内容

| 文件 | 说明 |
|------|------|
| `smartcar_predict.py` | 智能小车目标识别推理脚本 |

## 使用方法

```bash
python inference/smartcar_predict.py
```

推理脚本会加载项目根目录的 `smartcar_model.pth` 模型，对 `data/smartcar/test/` 目录下的测试图像进行预测并输出准确率。

## A4纸红色区域检测

A4纸红色区域检测功能位于 `preprocessing/detect_red.py`，用于检测图像中的红色标记区域并进行透视校正。

```bash
python preprocessing/detect_red.py
```
