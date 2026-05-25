"""Pipeline A: YOLOv8 detection, plate crop, ROI OCR, and region mapping."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from . import config
from .ocr_utils import extract_best_plate_candidate, run_paddleocr_robust, serialize_ocr_lines
from .region_mapper import extract_region_code, format_plate_text, identify_region_from_code
from .visualization import draw_yolo_bbox


def select_best_bbox_by_confidence(result: Any) -> dict[str, Any]:
    """Select highest-confidence YOLO bbox; area is the tie breaker."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return {"bbox": None, "confidence": None, "num_detections": 0, "status": "no_detection"}

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    order = np.lexsort((-areas, -confs))
    idx = int(order[0])
    status = "single_detection" if len(xyxy) == 1 else "multiple_detections_selected_highest_conf"
    return {
        "bbox": [float(v) for v in xyxy[idx]],
        "confidence": float(confs[idx]),
        "num_detections": int(len(xyxy)),
        "status": status,
    }


def crop_with_padding(image_bgr: np.ndarray, bbox_xyxy: list[float], padding_ratio: float) -> tuple[np.ndarray | None, list[int], str]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    x1_i = max(0, int(round(x1 - pad_x)))
    y1_i = max(0, int(round(y1 - pad_y)))
    x2_i = min(w, int(round(x2 + pad_x)))
    y2_i = min(h, int(round(y2 + pad_y)))
    if x2_i - x1_i < config.MIN_CROP_WIDTH or y2_i - y1_i < config.MIN_CROP_HEIGHT:
        return None, [x1_i, y1_i, x2_i, y2_i], "invalid_crop"
    return image_bgr[y1_i:y2_i, x1_i:x2_i].copy(), [x1_i, y1_i, x2_i, y2_i], "ok"


def extract_main_plate_roi(crop_bgr: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int], str]:
    if crop_bgr is None or crop_bgr.size == 0:
        return None, (0, 0, 0, 0), "no_crop"
    h, w = crop_bgr.shape[:2]
    x1 = max(0, int(round(config.MAIN_ROI_X_MIN * w)))
    x2 = min(w, int(round(config.MAIN_ROI_X_MAX * w)))
    y1 = max(0, int(round(config.MAIN_ROI_Y_MIN * h)))
    y2 = min(h, int(round(config.MAIN_ROI_Y_MAX * h)))
    if x2 <= x1 or y2 <= y1:
        return crop_bgr.copy(), (0, 0, w, h), "fallback_full_crop"
    return crop_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2), "ok"


def preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    image = image_bgr.copy()
    if config.OCR_UPSCALE and config.OCR_UPSCALE > 1:
        image = cv2.resize(image, None, fx=config.OCR_UPSCALE, fy=config.OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)
    if config.USE_CLAHE:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
        image = cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
    if config.USE_SHARPEN:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
    if config.USE_THRESHOLD:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5), cv2.COLOR_GRAY2BGR)
    return image


def run_pipeline_a(image_bgr: np.ndarray, yolo_model: Any, ocr_engine: Any, image_name: str = "uploaded_image") -> dict[str, Any]:
    """Run YOLO plate detection, OCR selected crop/ROI, and map Lampung region."""
    start = time.perf_counter()
    yolo_results = yolo_model.predict(
        source=image_bgr,
        conf=config.YOLO_CONF_THRES,
        iou=config.YOLO_IOU_THRES,
        verbose=False,
    )
    result = yolo_results[0]
    selected = select_best_bbox_by_confidence(result)
    visualized = draw_yolo_bbox(image_bgr, selected.get("bbox"), confidence=selected.get("confidence"))

    if selected.get("bbox") is None:
        return {
            "pipeline": "Pipeline A",
            "plate_pred": "",
            "plate_norm": "",
            "region_code": "",
            "region_name": "Unknown",
            "ocr_raw_text": "",
            "yolo_confidence": None,
            "num_detections": 0,
            "detection_status": "no_detection",
            "inference_time_ms": (time.perf_counter() - start) * 1000,
            "crop_image": None,
            "main_roi_image": None,
            "visualized_image": visualized,
            "notes": "YOLO tidak menemukan plat.",
        }

    crop, crop_box, crop_status = crop_with_padding(image_bgr, selected["bbox"], config.CROP_PADDING_RATIO)
    if crop is None:
        return {
            "pipeline": "Pipeline A",
            "plate_pred": "",
            "plate_norm": "",
            "region_code": "",
            "region_name": "Unknown",
            "ocr_raw_text": "",
            "yolo_confidence": selected.get("confidence"),
            "num_detections": selected.get("num_detections"),
            "detection_status": selected.get("status"),
            "inference_time_ms": (time.perf_counter() - start) * 1000,
            "crop_image": None,
            "main_roi_image": None,
            "visualized_image": visualized,
            "notes": f"Crop tidak valid: {crop_status}.",
        }

    main_roi, _, _ = extract_main_plate_roi(crop)
    enhanced_roi = preprocess_for_ocr(main_roi) if main_roi is not None else None
    sources = [
        run_paddleocr_robust(ocr_engine, crop, "full_crop", config.TEMP_DIR, image_name),
        run_paddleocr_robust(ocr_engine, main_roi, "main_roi", config.TEMP_DIR, image_name),
    ]
    # Enhanced ROI is useful only as fallback; scoring gives it a low source weight.
    sources.append(run_paddleocr_robust(ocr_engine, enhanced_roi, "enhanced_main_roi", config.TEMP_DIR, image_name))

    best = extract_best_plate_candidate(sources)
    plate_norm = best.get("plat_pred_norm", "")
    region_code = extract_region_code(plate_norm)
    raw_text = " | ".join(s.get("raw_text", "") for s in sources if s.get("raw_text"))
    ocr_statuses = ", ".join(f"{s.get('source')}={s.get('status')}" for s in sources)
    lines = []
    for source in sources:
        lines.extend(source.get("lines", []) or [])

    return {
        "pipeline": "Pipeline A",
        "plate_pred": format_plate_text(plate_norm) if plate_norm else "",
        "plate_norm": plate_norm,
        "region_code": region_code,
        "region_name": identify_region_from_code(region_code) if region_code else "Unknown",
        "ocr_raw_text": raw_text,
        "ocr_status": ocr_statuses,
        "ocr_lines_json": serialize_ocr_lines(lines),
        "selected_ocr_line": best.get("selected_raw_line", ""),
        "selected_candidate_score": best.get("candidate_score", 0.0),
        "candidate_rankings": best.get("all_candidates", []),
        "yolo_confidence": selected.get("confidence"),
        "num_detections": selected.get("num_detections"),
        "detection_status": selected.get("status"),
        "bbox_xyxy": selected.get("bbox"),
        "crop_box_xyxy": crop_box,
        "inference_time_ms": (time.perf_counter() - start) * 1000,
        "crop_image": crop,
        "main_roi_image": main_roi,
        "enhanced_roi_image": enhanced_roi,
        "visualized_image": visualized,
        "notes": best.get("candidate_filter_notes", ""),
    }
