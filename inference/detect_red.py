import cv2
import numpy as np
from pathlib import Path


def detect_a4_by_red(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([150, 100, 100])
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

    center_red_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        cnt_center_x = x + cw / 2

        if 0.25 * w < cnt_center_x < 0.75 * w and area > 1000 and cw > 40 and ch > 20:
            center_red_contours.append((cnt, area, x, y, cw, ch))

    if not center_red_contours:
        print(f"{image_path.name}: No center red region found")
        return img

    # === 原代码：按面积排序 ===
    center_red_contours.sort(key=lambda r: r[1], reverse=True)

    # 1. 取出最大面积的轮廓
    best_cnt = center_red_contours[0][0]

    # 为了方便计算，将轮廓点集从 (N, 1, 2) 展平为 (N, 2)
    # 此时 points 是一个 N行2列的数组，每行为 [x, y]
    points = best_cnt.reshape(-1, 2)

    # === 新的逻辑：寻找紧贴轮廓的四个数学极值点 ===

    # 计算 x + y 和 x - y
    add = points.sum(axis=1)  # N个点的 x+y 结果
    diff = np.diff(points, axis=1)  # N个点的 y-x 结果

    # 1. 左上角 (Top-Left): x + y 最小
    tl = points[np.argmin(add)]

    # 2. 右下角 (Bottom-Right): x + y 最大
    br = points[np.argmax(add)]

    # 3. 右上角 (Top-Right): x - y 最大  -> 相当于 y-x 最小
    tr = points[np.argmin(diff)]

    # 4. 左下角 (Bottom-Left): x - y 最小 -> 相当于 y-x 最大
    bl = points[np.argmax(diff)]

    # 将这四个角点整理成一个列表
    corner_points = [tl, tr, br, bl]

    # === 绘图部分 ===

    # 1. 依然画出蓝色的原轮廓线（参考线）
    cv2.drawContours(result, [best_cnt], -1, (255, 0, 0), 1)

    # 2. 遍历四个角点，画绿色实心圆
    for point in corner_points:
        # point 是 np.array([x, y])，需要转为元组 (x, y) 用于绘图
        px, py = point.ravel()
        # 画绿色 (0, 255, 0), 半径5, 实心 (-1) 的圆
        cv2.circle(result, (px, py), 1, (0, 255, 0), -1)

    # 3. 连接左边的上下两个点 (tl 和 bl)，延长到图像边界之外
    left_mid_x = (tl[0] + bl[0]) / 2
    left_mid_y = (tl[1] + bl[1]) / 2
    left_dir_x = tl[0] - bl[0]
    left_dir_y = tl[1] - bl[1]
    left_len = np.sqrt(left_dir_x**2 + left_dir_y**2)
    left_dir_x /= left_len
    left_dir_y /= left_len
    left_extend = 500
    left_pt1 = (
        int(left_mid_x - left_dir_x * left_extend),
        int(left_mid_y - left_dir_y * left_extend),
    )
    left_pt2 = (
        int(left_mid_x + left_dir_x * left_extend),
        int(left_mid_y + left_dir_y * left_extend),
    )
    cv2.line(result, left_pt1, left_pt2, (0, 255, 0), 1)

    # 4. 连接右边的上下两个点 (tr 和 br)，延长到图像边界之外
    right_mid_x = (tr[0] + br[0]) / 2
    right_mid_y = (tr[1] + br[1]) / 2
    right_dir_x = tr[0] - br[0]
    right_dir_y = tr[1] - br[1]
    right_len = np.sqrt(right_dir_x**2 + right_dir_y**2)
    right_dir_x /= right_len
    right_dir_y /= right_len
    right_extend = 500
    right_pt1 = (
        int(right_mid_x - right_dir_x * right_extend),
        int(right_mid_y - right_dir_y * right_extend),
    )
    right_pt2 = (
        int(right_mid_x + right_dir_x * right_extend),
        int(right_mid_y + right_dir_y * right_extend),
    )
    cv2.line(result, right_pt1, right_pt2, (0, 255, 0), 1)

    # --- 1. 整理红色区域已知的四个角点 (图像坐标) ---
    # 我们用 tl, tr, br, bl (你代码里已经算好的)
    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    # --- 2. 定义对应的物理坐标 (假设宽度为12cm，高度比例正确即可) ---
    # 红色区域在物理上是从 12cm 到 17cm 的位置
    w, h_img, h_red = 12.0, 12.0, 5.0
    dst_pts = np.array(
        [
            [0, h_img],  # 对应 tl (红色左上)
            [w, h_img],  # 对应 tr (红色右上)
            [w, h_img + h_red],  # 对应 br (红色右下)
            [0, h_img + h_red],  # 对应 bl (红色左下)
        ],
        dtype=np.float32,
    )

    # --- 3. 计算透视变换矩阵 M ---
    # 这个 M 记录了“物理平面”到“相机图片”的投影关系
    M = cv2.getPerspectiveTransform(dst_pts, src_pts)

    # --- 4. 用矩阵 M 推算纸张最顶部的两个点 ---
    # 纸张最顶部的物理坐标是 (0,0) 和 (12,0)
    top_points_physical = np.array([[[0, 0], [w, 0]]], dtype=np.float32)

    # 使用 cv2.perspectiveTransform 进行坐标变换
    top_points_image = cv2.perspectiveTransform(top_points_physical, M)[0]

    # --- 5. 绘图 ---
    pt_top_left = tuple(top_points_image[0].astype(int))
    pt_top_right = tuple(top_points_image[1].astype(int))

    # 连接顶边

    cv2.line(result, pt_top_left, pt_top_right, (0, 0, 255), 1)
    cv2.circle(result, pt_top_left, 2, (0, 0, 255), -1)
    cv2.circle(result, pt_top_right, 2, (0, 0, 255), -1)
    # 连接侧边补全
    cv2.line(result, tuple(tl.astype(int)), pt_top_left, (0, 255, 0), 1)
    cv2.line(result, tuple(tr.astype(int)), pt_top_right, (0, 255, 0), 1)

    src_quad = np.array([tl, tr, pt_top_right, pt_top_left], dtype=np.float32)
    width = int(np.linalg.norm(tr - tl))
    height = int(np.linalg.norm(pt_top_left - tl))
    dst_quad = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    M_warp = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(img, M_warp, (width, height))

    return result, warped


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
            output_data = detect_a4_by_red(img_path)
            if output_data is not None:
                result, warped = output_data
                output_path = category_out / img_path.name
                cv2.imwrite(str(output_path), result)
                warped_path = category_out/  f"warped_{img_path.name}"
                cv2.imwrite(str(warped_path), cv2.rotate(warped, cv2.ROTATE_180))


if __name__ == "__main__":
    main()
