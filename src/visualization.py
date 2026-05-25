"""Visualization helpers for Streamlit inference results."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def bgr_to_rgb(image_bgr: np.ndarray | None) -> np.ndarray | None:
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def draw_yolo_bbox(
    image_bgr: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float] | None,
    label: str = "license_plate",
    confidence: float | None = None,
) -> np.ndarray:
    canvas = image_bgr.copy()
    if bbox_xyxy is None:
        return canvas
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 210, 0), 3)
    caption = label
    if confidence is not None:
        caption = f"{label} {confidence:.2f}"
    cv2.putText(canvas, caption, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 0), 2, cv2.LINE_AA)
    return canvas


def draw_ocr_lines(
    image_bgr: np.ndarray,
    lines: list[dict[str, Any]],
    selected_raw_line: str = "",
) -> np.ndarray:
    canvas = image_bgr.copy()
    for line in lines or []:
        box = line.get("box")
        text = str(line.get("text", ""))
        if box is None:
            continue
        try:
            arr = np.asarray(box, dtype=float)
            if arr.ndim == 1 and arr.size >= 4:
                x1, y1, x2, y2 = [int(round(v)) for v in arr[:4]]
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            else:
                pts = arr.reshape(-1, 2).astype(np.int32)
            selected = bool(selected_raw_line and text == selected_raw_line)
            color = (0, 220, 0) if selected else (70, 150, 255)
            thickness = 3 if selected else 1
            cv2.polylines(canvas, [pts], True, color, thickness)
            x_text = int(np.min(pts[:, 0]))
            y_text = max(18, int(np.min(pts[:, 1])) - 4)
            cv2.putText(canvas, text[:28], (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        except Exception:
            continue
    return canvas

