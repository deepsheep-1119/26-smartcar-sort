# 智能小车目标识别

基于CNN的智能小车目标识别系统，支持三类目标检测：交通工具-直行、武器-左、物资-右。

## 依赖

```bash
uv sync
```

## 训练模型

```bash
python training/smartcar_train.py
```
训练完成后模型保存为 `smartcar_model.pth`

## 预测

```bash
python inference/smartcar_predict.py
```

## 图像预处理

```bash
python preprocessing/detect_red.py
```

## 数据集整理

```bash
python dataset/scripts/organize_dataset.py <源目录> --output <输出目录> --ratio <训练集比例>
```
