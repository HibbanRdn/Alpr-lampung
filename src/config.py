"""Application configuration for the ALPR Lampung demo."""

from __future__ import annotations

import os
from pathlib import Path


# PaddleOCR runtime flags must be set before Paddle/PaddleOCR is imported.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n" / "weights" / "best.pt"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_A_DIR = RESULTS_DIR / "pipeline_a"
RESULTS_B_DIR = RESULTS_DIR / "pipeline_b"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "assets" / "sample_images"
SCREENSHOTS_DIR = PROJECT_ROOT / "assets" / "screenshots"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CROPS_DIR = OUTPUT_DIR / "crops"
ROI_DIR = OUTPUT_DIR / "roi"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
TEMP_DIR = OUTPUT_DIR / "temp"

for directory in [OUTPUT_DIR, CROPS_DIR, ROI_DIR, VISUALIZATIONS_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES = 0.45
CROP_PADDING_RATIO = 0.05
MIN_CROP_WIDTH = 10
MIN_CROP_HEIGHT = 10

MAIN_ROI_Y_MIN = 0.05
MAIN_ROI_Y_MAX = 0.65
MAIN_ROI_X_MIN = 0.03
MAIN_ROI_X_MAX = 0.97

OCR_UPSCALE = 2
USE_CLAHE = True
USE_SHARPEN = False
USE_THRESHOLD = False

PADDLEOCR_KWARGS = {
    "lang": "en",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "text_detection_model_name": "PP-OCRv5_mobile_det",
    "text_recognition_model_name": "PP-OCRv5_mobile_rec",
}
