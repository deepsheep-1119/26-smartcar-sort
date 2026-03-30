import cv2
import numpy as np
from pathlib import Path


def detect_a4_by_red(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 120])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()

    if not contours:
        print(f"{image_path.name}: No red region found")
        return img

    valid_rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area > 2000 and w > 60 and h > 30:
            valid_rects.append((area, x, y, w, h))

    if not valid_rects:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        area = cv2.contourArea(largest)
        valid_rects.append((area, x, y, w, h))

    valid_rects.sort(key=lambda r: r[0], reverse=True)
    _, rx, ry, rw, rh = valid_rects[0]

    paper_h = int(rh * 4.0)
    paper_w = int(paper_h * 210 / 297)

    px = rx + rw // 2 - paper_w // 2
    py = ry - int(rh * 2.5)

    px = max(0, min(px, img.shape[1] - paper_w))
    py = max(0, min(py, img.shape[0] - paper_h))

    cv2.rectangle(result, (px, py), (px + paper_w, py + paper_h), (0, 255, 0), 3)
    print(f"{image_path.name}: Red({rx},{ry}) Paper({px},{py} {paper_w}x{paper_h})")

    return result


def main():
    input_dir = Path("png_smartcar")
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)

    categories = ["交通工具-直行", "武器-左", "物资-右"]

    for category in categories:
        category_path = input_dir / category
        if not category_path.exists():
            continue

        category_out = output_dir / category
        category_out.mkdir(exist_ok=True)

        for img_path in category_path.glob("*.png"):
            result = detect_a4_by_red(img_path)
            if result is not None:
                output_path = category_out / img_path.name
                cv2.imwrite(str(output_path), result)


if __name__ == "__main__":
    main()
