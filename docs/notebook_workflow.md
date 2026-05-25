# Workflow Notebook Project ALPR Lampung

Dokumen ini menjelaskan urutan eksperimen dari dataset mentah sampai aplikasi Streamlit. Streamlit tidak mengulang proses eksperimen; aplikasi hanya memakai model final, fungsi inference final, dan sebagian ringkasan hasil evaluasi.

## 1. Rename Dataset

Notebook:

```text
ALPR_Lampung_Dataset_Rename-2.ipynb
```

Tujuan:

- Merapikan nama file gambar awal.
- Mengubah nama gambar menjadi format konsisten seperti `alpr_lampung_000001.jpg`.
- Membuat nama file aman untuk dicocokkan dengan ground truth.

Output penting:

- Gambar dengan nama file konsisten.

Dipakai di Streamlit:

- Tidak langsung. Tahap ini hanya dokumentasi asal dataset.

## 2. Ground Truth Manual

File ground truth berisi:

```text
filename
plat_gt
kode_wilayah_gt
wilayah_gt
```

Tujuan:

- Menyimpan teks plat benar.
- Menyimpan kode dan wilayah Lampung untuk evaluasi OCR/wilayah.

Output penting:

- `ground_truth.csv`
- Setelah split final: `ground_truth_with_split.csv`

Dipakai di Streamlit:

- Tidak wajib untuk inference satu gambar.
- Opsional jika ingin membuat mode evaluasi sample.
- Ground truth tidak boleh dipakai untuk memilih prediksi.

## 3. Roboflow Annotation

Platform:

```text
Roboflow
```

Tujuan:

- Memberi anotasi objek `license_plate`.
- Membuat label lokasi plat untuk training YOLOv8.

Output penting:

- Dataset Roboflow export.
- Label awal berbentuk polygon/mask karena auto-label menggunakan SAM/masks.

Dipakai di Streamlit:

- Tidak. Streamlit hanya memakai model final hasil training.

## 4. Polygon to BBox Conversion

Notebook:

```text
alpr_lampung_colab_polygon_to_bbox-2.ipynb
```

Tujuan:

- Mengonversi label Roboflow polygon/mask menjadi bounding box YOLO detection.
- Format label akhir:

```text
class_id x_center y_center width height
```

Output penting:

- Dataset YOLO bbox yang siap untuk object detection.
- `conversion_report.csv`

Dipakai di Streamlit:

- Tidak langsung. Output tahap ini dipakai untuk training YOLO, bukan untuk runtime aplikasi.

## 5. Final Split Dataset

Notebook:

```text
alpr_lampung_final_split_colab-3.ipynb
```

Tujuan:

- Membersihkan nama file Roboflow yang memiliki suffix random.
- Mencocokkan file dengan ground truth.
- Membuat split akhir train/valid/test.

Output penting:

```text
ALPR-Lampung-final/
├── data.yaml
├── ground_truth_with_split.csv
├── train/images
├── valid/images
└── test/images
```

Dipakai di Streamlit:

- Tidak seluruh folder dataset.
- Hanya beberapa sample image dari `test/images` yang boleh disalin ke `assets/sample_images/`.
- `ground_truth_with_split.csv` opsional untuk dokumentasi/evaluasi sample.

## 6. YOLOv8 Training

Notebook:

```text
alpr_lampung_yolov8_training_colab-3.ipynb
```

Tujuan:

- Training YOLOv8n untuk mendeteksi satu class: `license_plate`.
- Evaluasi deteksi dengan precision, recall, mAP@0.5, dan mAP@0.5:0.95.

Output penting:

```text
/content/drive/MyDrive/ALPR-Lampung-final/models/yolov8n/weights/best.pt
```

Dipakai di Streamlit:

- Ya. Salin ke:

```text
models/yolov8n/weights/best.pt
```

Catatan:

- Streamlit tidak melakukan training ulang.

## 7. Pipeline A Evaluation

Notebook:

```text
alpr_lampung_pipeline_a_v2_line_scoring_colab-2.ipynb
```

Tujuan:

- YOLOv8 mendeteksi plat.
- Crop plat dan ambil ROI teks utama.
- PaddleOCR membaca crop/ROI.
- Candidate extraction memilih teks plat utama.
- Region mapping menentukan wilayah Lampung.
- Evaluasi terhadap ground truth.

Output penting:

```text
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_a/hasil_pipeline_a.csv
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_a/ringkasan_evaluasi_pipeline_a.csv
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_a/dashboard_ringkas_pipeline_a.csv
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_a/technical_outputs/
```

Dipakai di Streamlit:

- Logic inference dipindahkan ke:

```text
src/pipeline_a.py
src/ocr_utils.py
src/region_mapper.py
src/visualization.py
```

- Ringkasan evaluasi dapat disalin ke:

```text
results/pipeline_a/
```

## 8. Pipeline B Evaluation

Notebook:

```text
alpr_lampung_pipeline_b_paddleocr_full_image_colab-2.ipynb
```

Tujuan:

- PaddleOCR langsung membaca full image.
- Candidate extraction memilih teks paling mirip plat Lampung.
- Region mapping dan evaluasi.

Pipeline B tidak memakai YOLO dan tidak memakai crop.

Output penting:

```text
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_b/hasil_pipeline_b.csv
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_b/ringkasan_evaluasi_pipeline_b.csv
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_b/perbandingan_pipeline_a_vs_b.csv
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_b/technical_outputs/
```

Dipakai di Streamlit:

- Logic inference dipindahkan ke:

```text
src/pipeline_b.py
src/ocr_utils.py
src/region_mapper.py
src/visualization.py
```

- Ringkasan evaluasi dapat disalin ke:

```text
results/pipeline_b/
```

## 9. Streamlit Deployment

File utama:

```text
app.py
```

Tujuan:

- Upload gambar kendaraan.
- Pilih Pipeline A, Pipeline B, atau Compare A vs B.
- Menampilkan prediksi plat dan wilayah.
- Menampilkan OCR mentah, confidence, visualisasi, dan detail teknis.
- Menyediakan input manual plat untuk mengecek mapping wilayah.

Output runtime:

```text
outputs/crops/
outputs/roi/
outputs/visualizations/
outputs/temp/
```

Dipakai di Streamlit:

- Ya, sebagai aplikasi demo/inference.

Streamlit tidak melakukan:

- rename dataset,
- konversi polygon ke bbox,
- split dataset,
- training YOLO,
- evaluasi batch terhadap ground truth.

