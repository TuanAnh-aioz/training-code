from typing import Tuple

import cv2
import numpy as np
from matplotlib import cm


def get_label_colors(class_to_idx: dict):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    cmap_name = "tab20" if num_classes <= 20 else "gist_ncar"
    cmap = cm.get_cmap(cmap_name, num_classes)

    colors = {}
    for idx in sorted(idx_to_class.keys()):
        if idx == 0:
            colors[idx] = (0, 0, 0)
        else:
            color = cmap(idx % num_classes)
            rgb = tuple(int(255 * c) for c in color[:3])
            colors[idx] = rgb

    return colors


def draw_transparent_box(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], alpha: float = 0.3):
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if img is None or img.ndim != 3:
        raise ValueError("Input image must be a valid 3-channel image.")

    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_label(img: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int], font_scale: float = 0.5, thickness: int = 1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y_text = max(y, th + 4)

    cv2.rectangle(img, (x, y_text - th - 4), (x + tw, y_text + baseline), color, -1)

    brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

    # Put text
    cv2.putText(img, text, (x, y_text - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_rounded_rectangle(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2, r: int = 10):
    x1, y1 = pt1
    x2, y2 = pt2

    r = min(r, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    # 4 lines
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # 4 corner arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
