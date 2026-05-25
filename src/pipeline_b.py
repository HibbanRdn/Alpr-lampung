"""Pipeline B: full image PaddleOCR without YOLO."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from . import config
from .ocr_utils import extract_best_plate_candidate, run_paddleocr_robust, serialize_ocr_lines
from .region_mapper import extract_region_code, format_plate_text, identify_region_from_code
from .visualization import draw_ocr_lines


def run_pipeline_b(image_bgr: np.ndarray, ocr_engine: Any, image_name: str = "uploaded_image") -> dict[str, Any]:
    """Run full-image OCR and pick the best Lampung plate candidate."""
    start = time.perf_counter()
    ocr_result = run_paddleocr_robust(
        ocr_engine=ocr_engine,
        image=image_bgr,
        source_name="full_image",
        temp_dir=config.TEMP_DIR,
        image_name=image_name,
    )

    raw_text = ocr_result.get("raw_text", "")
    lines = ocr_result.get("lines", []) or []
    best = extract_best_plate_candidate([ocr_result]) if raw_text.strip() else {
        "plat_pred_norm": "",
        "plat_pred_formatted": "",
        "selected_raw_line": "",
        "candidate_score": 0.0,
        "candidate_filter_notes": "OCR kosong atau error",
        "all_candidates": [],
    }

    plate_norm = best.get("plat_pred_norm", "")
    region_code = extract_region_code(plate_norm)
    visualized = draw_ocr_lines(image_bgr, lines, best.get("selected_raw_line", ""))

    return {
        "pipeline": "Pipeline B",
        "plate_pred": format_plate_text(plate_norm) if plate_norm else "",
        "plate_norm": plate_norm,
        "region_code": region_code,
        "region_name": identify_region_from_code(region_code) if region_code else "Unknown",
        "ocr_raw_text": raw_text,
        "ocr_status": ocr_result.get("status", ""),
        "ocr_mode_used": ocr_result.get("mode_used", ""),
        "ocr_lines": lines,
        "ocr_lines_json": serialize_ocr_lines(lines),
        "selected_ocr_line": best.get("selected_raw_line", ""),
        "selected_candidate_score": best.get("candidate_score", 0.0),
        "candidate_rankings": best.get("all_candidates", []),
        "inference_time_ms": (time.perf_counter() - start) * 1000,
        "visualized_image": visualized,
        "notes": best.get("candidate_filter_notes", "") or ocr_result.get("error", ""),
    }
