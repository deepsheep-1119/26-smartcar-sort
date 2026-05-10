import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import tensorflow as tf
from PIL import Image


def predict_image(interpreter, img_path, idx_to_class):
    img = Image.open(str(img_path)).convert("RGB")
    img = img.resize((96, 96))
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    # 获取输入输出张量
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 设置输入
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # 运行推理
    interpreter.invoke()

    # 获取输出
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    probs = tf.nn.softmax(output).numpy()
    pred = int(np.argmax(output))

    print(f"置信度:")
    for i, cls_name in idx_to_class.items():
        print(f"  {cls_name}: {probs[int(i)]:.4f}")

    return idx_to_class[str(pred)]


def main():
    if sys.argv!=2:
        print("argv shoule == 2")
    
    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_path="smartcar_model.tflite")
    interpreter.allocate_tensors()

    with open("idx_to_class_tf.json", "r") as f:
        idx_to_class = json.load(f)

    print(f"Classes: {idx_to_class}")
    
    pred = predict_image(interpreter, sys.argv[1], idx_to_class)
    print(f"{sys.argv[1]}: predicted={pred}")


if __name__ == "__main__":
    main()