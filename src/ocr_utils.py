"""PaddleOCR parsing, robust OCR execution, and plate candidate scoring."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .region_mapper import format_plate_text, normalize_plate_text


OCR_MODES = ("predict_path", "predict_bgr", "predict_rgb", "legacy_ocr_path", "legacy_ocr_bgr")
NON_PLATE_WORDS = {
    "HONDA",
    "YAMAHA",
    "TOYOTA",
    "DAIHATSU",
    "SUZUKI",
    "MITSUBISHI",
    "NISSAN",
    "ONEHEART",
    "ONE",
    "HEART",
    "VARIO",
    "BEAT",
    "AVANZA",
    "INNOVA",
    "BRIO",
    "CIVIC",
    "JAZZ",
    "MOBIL",
    "MOTOR",
}
COMMON_TWO_LETTER_SUFFIXES = {"FL", "FT", "FR", "GP", "BW", "RG", "CF", "HX", "KK"}


def _box_to_xy_stats(box: Any, shape: tuple[int, int] | tuple[int, int, int] | None) -> dict[str, Any]:
    if box is None or shape is None:
        return {}
    h, w = shape[:2]
    try:
        arr = np.asarray(box, dtype=float)
        if arr.size == 0:
            return {}
        if arr.ndim == 1 and arr.size >= 4:
            xs = np.array([arr[0], arr[2]], dtype=float)
            ys = np.array([arr[1], arr[3]], dtype=float)
        elif arr.ndim >= 2 and arr.shape[-1] >= 2:
            xs = arr[..., 0].reshape(-1)
            ys = arr[..., 1].reshape(-1)
        else:
            return {}
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "center_x_ratio": ((x_min + x_max) / 2) / max(1, w),
            "center_y_ratio": ((y_min + y_max) / 2) / max(1, h),
            "width_ratio": (x_max - x_min) / max(1, w),
            "height_ratio": (y_max - y_min) / max(1, h),
        }
    except Exception:
        return {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()]
        return [item for item in value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _first_present(obj: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return None


def _get_payload(obj: Any, attr_name: str) -> Any:
    if not hasattr(obj, attr_name):
        return None
    try:
        attr = getattr(obj, attr_name)
        return attr() if callable(attr) else attr
    except Exception:
        return None


def _append_lines_from_dict(obj: dict[str, Any], lines: list[dict[str, Any]], image_shape: Any) -> None:
    texts = _as_list(_first_present(obj, ["rec_texts", "texts", "text"]))
    scores = _as_list(_first_present(obj, ["rec_scores", "scores", "score", "confidence"]))
    boxes = _as_list(_first_present(obj, ["rec_boxes", "rec_polys", "dt_polys", "boxes", "polys"]))
    if not texts:
        return
    for idx, text in enumerate(texts):
        text = str(text).strip()
        if not text:
            continue
        score = None
        if idx < len(scores):
            try:
                score = float(scores[idx])
            except Exception:
                score = None
        box = boxes[idx] if idx < len(boxes) else None
        lines.append({"text": text, "score": score, "box": box, **_box_to_xy_stats(box, image_shape)})


def parse_paddleocr_output(output: Any, image_shape: Any = None) -> tuple[str, float | None, list[dict[str, Any]], str, str]:
    """Parse PaddleOCR v3 and legacy outputs into text, mean score, and OCR lines."""
    lines: list[dict[str, Any]] = []
    visited: set[int] = set()

    def collect(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, dict):
            _append_lines_from_dict(obj, lines, image_shape)
            for value in obj.values():
                if isinstance(value, (dict, list, tuple)) or hasattr(value, "json") or hasattr(value, "res"):
                    collect(value)
            return

        for attr in ("json", "res", "to_dict", "as_dict"):
            payload = _get_payload(obj, attr)
            if payload is not None:
                collect(payload)

        if isinstance(obj, (list, tuple)):
            if len(obj) >= 2 and isinstance(obj[1], (tuple, list)) and len(obj[1]) >= 2:
                maybe_text = obj[1][0]
                if isinstance(maybe_text, str) and maybe_text.strip():
                    try:
                        score = float(obj[1][1])
                    except Exception:
                        score = None
                    lines.append({"text": maybe_text.strip(), "score": score, "box": obj[0], **_box_to_xy_stats(obj[0], image_shape)})
                    return
            for item in obj:
                collect(item)

    try:
        collect(output)
    except Exception as exc:
        return "", None, [], "parser_error", str(exc)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line in lines:
        key = (line.get("text", ""), str(line.get("box", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)

    text = " ".join(line["text"] for line in deduped if line.get("text"))
    scores = [float(line["score"]) for line in deduped if line.get("score") is not None]
    confidence = float(np.mean(scores)) if scores else None
    parse_warning = "raw output exists but parser extracted no text" if output is not None and not text else ""
    return text, confidence, deduped, "ok", parse_warning


def serialize_ocr_lines(lines: list[dict[str, Any]]) -> str:
    """Serialize OCR lines for technical display."""
    safe_lines = []
    for line in lines or []:
        safe = dict(line)
        if safe.get("box") is not None:
            try:
                safe["box"] = np.asarray(safe["box"]).tolist()
            except Exception:
                safe["box"] = str(safe["box"])
        safe_lines.append(safe)
    return json.dumps(safe_lines, ensure_ascii=False)


def prepare_ocr_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _safe_type_name(obj: Any) -> str:
    try:
        return f"{type(obj).__module__}.{type(obj).__name__}"
    except Exception:
        return str(type(obj))


def run_paddleocr_robust(
    ocr_engine: Any,
    image: np.ndarray | None,
    source_name: str,
    temp_dir: Path,
    image_name: str = "image",
) -> dict[str, Any]:
    """Run PaddleOCR with several input modes and return the first useful text."""
    image_bgr = prepare_ocr_image(image)
    base = {
        "source": source_name,
        "raw_text": "",
        "confidence": None,
        "status": "no_image" if image_bgr is None else "not_started",
        "error": "",
        "mode_used": "",
        "lines": [],
        "parse_warning": "",
        "raw_output_type": "",
    }
    if image_bgr is None:
        return base

    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(image_name).stem or "image")
    temp_path = temp_dir / f"{safe_stem}_{source_name}.jpg"
    if not cv2.imwrite(str(temp_path), image_bgr):
        base.update({"status": "ocr_error", "error": f"gagal menulis temp file: {temp_path}"})
        return base

    attempts: list[dict[str, Any]] = []
    best_empty: dict[str, Any] | None = None
    for mode in OCR_MODES:
        output = None
        try:
            if mode == "predict_path":
                output = ocr_engine.predict(str(temp_path))
            elif mode == "predict_bgr":
                output = ocr_engine.predict(image_bgr)
            elif mode == "predict_rgb":
                output = ocr_engine.predict(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            elif mode == "legacy_ocr_path":
                output = ocr_engine.ocr(str(temp_path))
            elif mode == "legacy_ocr_bgr":
                output = ocr_engine.ocr(image_bgr)
            parsed_text, parsed_conf, lines, parse_status, parse_warning = parse_paddleocr_output(output, image_bgr.shape)
            attempt = {
                "mode": mode,
                "status": "ok" if parsed_text.strip() else "ocr_empty",
                "raw_text": parsed_text,
                "confidence": parsed_conf,
                "lines": lines,
                "parse_status": parse_status,
                "parse_warning": parse_warning,
                "raw_output_type": _safe_type_name(output),
                "error": "",
            }
            attempts.append(attempt)
            if parsed_text.strip():
                return {**base, **attempt, "mode_used": mode}
            best_empty = best_empty or attempt
        except Exception as exc:
            attempts.append({"mode": mode, "status": "ocr_error", "error": str(exc), "lines": []})

    if best_empty is not None:
        return {**base, **best_empty, "mode_used": best_empty["mode"]}
    error = " | ".join(f"{a['mode']}: {a.get('error', '')}" for a in attempts if a.get("status") == "ocr_error")
    return {**base, "status": "ocr_error", "error": error, "attempts": attempts}


def _geometry_score(line: dict[str, Any] | None) -> float:
    if not line:
        return -4.0
    w = line.get("width_ratio")
    h = line.get("height_ratio")
    cy = line.get("center_y_ratio")
    score = 0.0
    if w is not None and h is not None and h > 0:
        aspect = w / max(h, 1e-6)
        score += 6.0 if aspect >= 2.0 else -5.0 if aspect < 1.0 else 0.0
        score += 4.0 if 0.035 <= w <= 0.95 else -8.0 if w < 0.02 else 0.0
        score += 4.0 if 0.008 <= h <= 0.18 else -8.0 if h < 0.006 else 0.0
    if cy is not None:
        score += 2.0 if 0.05 <= cy <= 0.95 else -4.0
    return score


def _noise_penalty(raw_text: str, trailing_noise: str, source_kind: str) -> tuple[float, list[str]]:
    notes: list[str] = []
    penalty = 0.0
    raw_clean = normalize_plate_text(raw_text)
    words = re.findall(r"[A-Z]+", str(raw_text).upper())
    hits = [word for word in words if word in NON_PLATE_WORDS]
    if hits:
        penalty -= 8.0 * len(hits)
        notes.append(f"teks non-plat terdeteksi: {','.join(hits)}")
    if trailing_noise:
        notes.append(f"noise belakang dibuang: {trailing_noise}")
        penalty -= 1.0 if re.search(r"\d{2,4}", trailing_noise) else min(len(trailing_noise), 12) * 0.7
    if len(raw_clean) > 14 and source_kind.endswith("_joined"):
        penalty -= 8.0
        notes.append("gabungan teks terlalu panjang")
    if re.fullmatch(r"\d{2,4}", raw_clean):
        penalty -= 30.0
        notes.append("line angka saja kemungkinan tanggal")
    return penalty, notes


def _generate_candidates_from_text_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    source = item.get("source", "ocr")
    source_kind = item.get("source_kind", "line")
    raw_line = str(item.get("raw_line", item.get("text", "")))
    source_text_all = str(item.get("source_text_all", raw_line))
    confidence = item.get("confidence")
    line = item.get("line")
    line_index = item.get("line_index")
    clean = normalize_plate_text(item.get("text", ""))
    candidates: list[dict[str, Any]] = []

    for match in re.finditer(r"BE([0-9]{1,4})([A-Z0-9]{1,12})", clean):
        digits = match.group(1)
        tail = match.group(2)
        letter_match = re.match(r"([A-Z]{1,6})(.*)", tail)
        if not letter_match:
            continue
        letter_run = letter_match.group(1)
        after_letter_run = letter_match.group(2)
        max_suffix_len = min(3, len(letter_run))
        for suffix_len in range(1, max_suffix_len + 1):
            suffix = letter_run[:suffix_len]
            trailing_noise = letter_run[suffix_len:] + after_letter_run + clean[match.end() :]
            candidate_norm = f"BE{digits}{suffix}"
            if not re.fullmatch(r"BE\d{1,4}[A-Z]{1,3}", candidate_norm):
                continue

            pattern_score = 100.0 + suffix_len * 2.0
            if suffix_len == 3:
                pattern_score += 4.0
            if not trailing_noise:
                pattern_score += 8.0
            elif re.search(r"\d{2,4}", trailing_noise):
                pattern_score += 2.0
            if suffix_len == 3 and suffix[:2] in COMMON_TWO_LETTER_SUFFIXES and trailing_noise:
                pattern_score -= 18.0
            if suffix_len < 3 and len(letter_run) >= 3 and suffix[:2] not in COMMON_TWO_LETTER_SUFFIXES:
                pattern_score -= 3.0

            confidence_bonus = 0.0 if confidence is None else float(confidence) * 14.0
            line_bonus = 10.0 if source_kind.endswith("_line") else -8.0
            if source_kind == "main_roi_line":
                line_bonus += 2.0
            elif source_kind == "enhanced_main_roi_line":
                line_bonus -= 6.0
            geometry_score = _geometry_score(line) if source_kind.endswith("_line") else -4.0
            noise_penalty, notes = _noise_penalty(raw_line, trailing_noise, source_kind)
            final_score = pattern_score + confidence_bonus + line_bonus + geometry_score + noise_penalty

            candidates.append(
                {
                    "source": source,
                    "source_kind": source_kind,
                    "line_index": line_index,
                    "raw_line": raw_line,
                    "source_text_all": source_text_all,
                    "candidate_norm": candidate_norm,
                    "candidate_formatted": format_plate_text(candidate_norm),
                    "ocr_confidence": confidence,
                    "pattern_score": pattern_score,
                    "geometry_score": geometry_score,
                    "noise_penalty": noise_penalty,
                    "line_bonus": line_bonus,
                    "final_score": final_score,
                    "removed_noise_suffix": trailing_noise,
                    "notes": "; ".join(notes) if notes else "candidate dari OCR line",
                    "line": line,
                }
            )
    return candidates


def build_candidate_inputs_from_ocr_results(ocr_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for result in ocr_results:
        source = result.get("source", "ocr")
        raw_text = result.get("raw_text", "")
        lines = result.get("lines", []) or []
        for idx, line in enumerate(lines):
            text = str(line.get("text", "")).strip()
            if not text:
                continue
            inputs.append(
                {
                    "source": source,
                    "source_kind": f"{source}_line",
                    "raw_line": text,
                    "text": text,
                    "source_text_all": raw_text,
                    "confidence": line.get("score"),
                    "line_index": idx,
                    "line": line,
                }
            )
        if raw_text.strip():
            inputs.append(
                {
                    "source": source,
                    "source_kind": f"{source}_joined",
                    "raw_line": raw_text,
                    "text": raw_text,
                    "source_text_all": raw_text,
                    "confidence": result.get("confidence"),
                    "line_index": None,
                    "line": None,
                }
            )
    return inputs


def extract_best_plate_candidate(ocr_results_or_text: list[dict[str, Any]] | str) -> dict[str, Any]:
    """Select the best BE plate candidate from OCR lines or raw text."""
    if isinstance(ocr_results_or_text, str):
        ocr_results = [{"source": "manual", "raw_text": ocr_results_or_text, "confidence": None, "lines": [{"text": ocr_results_or_text, "score": None}]}]
    else:
        ocr_results = ocr_results_or_text

    inputs = build_candidate_inputs_from_ocr_results(ocr_results)
    candidates: list[dict[str, Any]] = []
    for item in inputs:
        candidates.extend(_generate_candidates_from_text_item(item))

    if not candidates:
        return {
            "plat_pred_norm": "",
            "plat_pred_formatted": "",
            "selected_raw_line": "",
            "selected_candidate_text": "",
            "candidate_score": 0.0,
            "all_candidates": [],
            "candidate_filter_notes": "tidak ada kandidat plat BE yang valid",
        }

    # Agreement bonus: candidates repeated by multiple OCR sources become stronger.
    counts: dict[str, int] = {}
    for cand in candidates:
        counts[cand["candidate_norm"]] = counts.get(cand["candidate_norm"], 0) + 1
    for cand in candidates:
        cand["agreement_bonus"] = max(0, counts[cand["candidate_norm"]] - 1) * 8.0
        cand["final_score"] += cand["agreement_bonus"]

    candidates = sorted(candidates, key=lambda c: c.get("final_score", 0), reverse=True)
    best = candidates[0]
    return {
        "plat_pred_norm": best["candidate_norm"],
        "plat_pred_formatted": best["candidate_formatted"],
        "selected_raw_line": best["raw_line"],
        "selected_candidate_text": best["candidate_formatted"],
        "candidate_score": float(best.get("final_score", 0.0)),
        "all_candidates": candidates,
        "candidate_filter_notes": best.get("notes", ""),
        "removed_noise_suffix": best.get("removed_noise_suffix", ""),
    }

