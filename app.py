from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import config
from src.pipeline_a import run_pipeline_a
from src.pipeline_b import run_pipeline_b
from src.region_mapper import map_plate_to_region
from src.visualization import bgr_to_rgb


st.set_page_config(
    page_title="ALPR Lampung Dashboard",
    layout="wide",
)


@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def load_yolo_model():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ultralytics/YOLO belum terinstall di runtime ini. "
            "Untuk memakai Pipeline A, gunakan Python 3.12 lalu install dependency ML dari requirements.txt."
        ) from exc

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model YOLO tidak ditemukan di {config.MODEL_PATH}. "
            "Letakkan best.pt hasil training ke folder models/yolov8n/weights/."
        )
    return YOLO(str(config.MODEL_PATH))


@st.cache_resource(show_spinner="Memuat PaddleOCR...")
def load_ocr_engine():
    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PaddleOCR belum terinstall di runtime ini. "
            "Streamlit Cloud yang berjalan di Python 3.14 tidak kompatibel dengan paddlepaddle==3.2.2. "
            "Atur Python version ke 3.12 di Manage app > Settings agar Pipeline A/B dapat berjalan penuh."
        ) from exc

    try:
        return PaddleOCR(**config.PADDLEOCR_KWARGS)
    except TypeError:
        return PaddleOCR(lang="en")


def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def display_prediction_card(result: dict, title: str) -> None:
    st.subheader(title)
    cols = st.columns(4)
    cols[0].metric("Plat Terdeteksi", result.get("plate_pred") or "-")
    cols[1].metric("Kode Wilayah", result.get("region_code") or "-")
    cols[2].metric("Wilayah Prediksi", result.get("region_name") or "-")
    cols[3].metric("Waktu Proses", f"{result.get('inference_time_ms', 0) / 1000:.2f} detik")

    if result.get("yolo_confidence") is not None:
        st.caption(f"Confidence Deteksi YOLO: {result['yolo_confidence']:.3f}")
    if result.get("notes"):
        st.info(result["notes"])

    st.write("**Teks OCR Mentah**")
    st.code(result.get("ocr_raw_text") or "-", language="text")

    with st.expander("Detail Teknis"):
        detail = {
            "Pipeline": result.get("pipeline"),
            "Plate normalized": result.get("plate_norm"),
            "Selected OCR line": result.get("selected_ocr_line", ""),
            "Candidate score": result.get("selected_candidate_score", ""),
            "Status OCR": result.get("ocr_status", ""),
            "Mode OCR": result.get("ocr_mode_used", ""),
            "Jumlah deteksi YOLO": result.get("num_detections", ""),
            "Status deteksi": result.get("detection_status", ""),
            "BBox": result.get("bbox_xyxy", ""),
        }
        st.json(detail)
        candidates = result.get("candidate_rankings") or []
        if candidates:
            candidate_rows = []
            for cand in candidates[:10]:
                candidate_rows.append(
                    {
                        "source": cand.get("source"),
                        "raw_line": cand.get("raw_line"),
                        "candidate": cand.get("candidate_formatted"),
                        "score": cand.get("final_score"),
                        "notes": cand.get("notes"),
                    }
                )
            st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True)


def display_pipeline_a(result: dict) -> None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(bgr_to_rgb(result.get("visualized_image")), caption="Gambar dengan bbox YOLO", use_container_width=True)
    with col2:
        if result.get("crop_image") is not None:
            st.image(bgr_to_rgb(result["crop_image"]), caption="Crop Plat", use_container_width=True)
        if result.get("main_roi_image") is not None:
            st.image(bgr_to_rgb(result["main_roi_image"]), caption="ROI Teks Utama", use_container_width=True)
    display_prediction_card(result, "Hasil Pipeline A")


def display_pipeline_b(result: dict) -> None:
    st.image(bgr_to_rgb(result.get("visualized_image")), caption="Full image dengan OCR line terpilih", use_container_width=True)
    display_prediction_card(result, "Hasil Pipeline B")


def run_selected_pipeline(mode: str, image_bgr: np.ndarray, image_name: str) -> None:
    try:
        ocr_engine = load_ocr_engine()
    except RuntimeError as exc:
        st.error(str(exc))
        st.info(
            "App berhasil deploy, tetapi fitur OCR membutuhkan PaddleOCR. "
            "Jika ingin menjalankan inferensi penuh di Streamlit Cloud, ubah Python version app ke 3.12 lalu redeploy."
        )
        return

    if mode == "Pipeline A: YOLOv8 + PaddleOCR":
        try:
            yolo_model = load_yolo_model()
        except RuntimeError as exc:
            st.error(str(exc))
            return
        result_a = run_pipeline_a(image_bgr, yolo_model, ocr_engine, image_name=image_name)
        display_pipeline_a(result_a)
        return

    if mode == "Pipeline B: Full Image PaddleOCR":
        result_b = run_pipeline_b(image_bgr, ocr_engine, image_name=image_name)
        display_pipeline_b(result_b)
        return

    try:
        yolo_model = load_yolo_model()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    result_a = run_pipeline_a(image_bgr, yolo_model, ocr_engine, image_name=image_name)
    result_b = run_pipeline_b(image_bgr, ocr_engine, image_name=image_name)

    comparison = pd.DataFrame(
        [
            {
                "Pipeline": "Pipeline A",
                "Plat Prediksi": result_a.get("plate_pred") or "-",
                "Wilayah": result_a.get("region_name") or "-",
                "OCR Raw": result_a.get("ocr_raw_text") or "-",
                "Waktu Proses": f"{result_a.get('inference_time_ms', 0) / 1000:.2f} detik",
                "Catatan": result_a.get("notes", ""),
            },
            {
                "Pipeline": "Pipeline B",
                "Plat Prediksi": result_b.get("plate_pred") or "-",
                "Wilayah": result_b.get("region_name") or "-",
                "OCR Raw": result_b.get("ocr_raw_text") or "-",
                "Waktu Proses": f"{result_b.get('inference_time_ms', 0) / 1000:.2f} detik",
                "Catatan": result_b.get("notes", ""),
            },
        ]
    )
    st.subheader("Perbandingan Pipeline A vs Pipeline B")
    st.dataframe(comparison, use_container_width=True)
    tab_a, tab_b = st.tabs(["Visual Pipeline A", "Visual Pipeline B"])
    with tab_a:
        display_pipeline_a(result_a)
    with tab_b:
        display_pipeline_b(result_b)


st.title("Dashboard ALPR Lampung")
st.write(
    "Aplikasi demo ini membaca plat kendaraan Lampung dengan dua mode: "
    "Pipeline A memakai YOLOv8 untuk mendeteksi plat terlebih dahulu, sedangkan Pipeline B menjalankan PaddleOCR langsung pada gambar penuh."
)

missing_runtime_notes = []
if importlib.util.find_spec("paddleocr") is None:
    missing_runtime_notes.append("PaddleOCR")
if importlib.util.find_spec("ultralytics") is None:
    missing_runtime_notes.append("Ultralytics/YOLO")
if missing_runtime_notes:
    st.warning(
        "Runtime saat ini belum memuat dependency ML: "
        + ", ".join(missing_runtime_notes)
        + ". UI tetap bisa dibuka, tetapi inferensi penuh membutuhkan Python 3.12 di Streamlit Cloud."
    )
    with st.expander("Cara mengaktifkan inferensi penuh di Streamlit Cloud"):
        st.write("1. Buka **Manage app** di Streamlit Community Cloud.")
        st.write("2. Masuk ke **Settings** atau **Advanced settings**.")
        st.write("3. Ubah **Python version** menjadi **3.12**.")
        st.write("4. Klik **Reboot** atau **Redeploy**.")
        st.caption(f"Python runtime saat ini: {sys.version.split()[0]}")

with st.sidebar:
    st.header("Pengaturan")
    mode = st.radio(
        "Pilih mode inferensi",
        [
            "Pipeline A: YOLOv8 + PaddleOCR",
            "Pipeline B: Full Image PaddleOCR",
            "Compare A vs B",
        ],
    )
    st.caption("Aplikasi ini hanya inference, bukan training.")

    st.divider()
    st.subheader("Cek Wilayah Manual")
    manual_plate = st.text_input("Masukkan plat manual", placeholder="Contoh: BE 1193 ALQ")
    if manual_plate:
        mapped = map_plate_to_region(manual_plate)
        st.write("Plat normalized:", mapped["plate_norm"] or "-")
        st.write("Kode wilayah:", mapped["region_code"] or "-")
        st.write("Wilayah:", mapped["region_name"])
        if mapped["is_valid"]:
            st.success("Format plat valid.")
        else:
            st.warning("Format belum cocok dengan pola BE + angka + huruf belakang.")

uploaded = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Silakan upload gambar kendaraan untuk memulai inferensi.")
    sample_dir = config.PROJECT_ROOT / "assets" / "sample_images"
    sample_images = sorted(sample_dir.glob("*.jpg"))
    if sample_images:
        st.write("Contoh gambar tersedia di folder `assets/sample_images/`.")
    st.stop()

image_bgr = uploaded_file_to_bgr(uploaded)
st.subheader("Preview Gambar Input")
st.image(bgr_to_rgb(image_bgr), caption=uploaded.name, use_container_width=True)

if st.button("Run Inference", type="primary"):
    try:
        with st.spinner("Menjalankan inferensi..."):
            run_selected_pipeline(mode, image_bgr, uploaded.name)
    except Exception as exc:
        st.error("Inferensi gagal.")
        st.exception(exc)
