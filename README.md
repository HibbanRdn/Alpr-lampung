# ALPR Lampung Streamlit Dashboard

Dashboard ini adalah aplikasi demo/inference untuk project **ALPR Lampung License Plate Detection**. Aplikasi menerima gambar kendaraan, lalu menjalankan salah satu pipeline:

1. **Pipeline A: YOLOv8 + PaddleOCR**  
   YOLOv8 mendeteksi lokasi plat, aplikasi mengambil crop/ROI plat, PaddleOCR membaca teks, lalu kode belakang plat dipetakan ke wilayah Lampung.

2. **Pipeline B: Full Image PaddleOCR**  
   PaddleOCR langsung membaca gambar penuh tanpa YOLO dan tanpa crop. Pipeline ini menjadi baseline pembanding Pipeline A.

3. **Compare A vs B**  
   Menjalankan kedua pipeline pada gambar yang sama dan menampilkan perbandingan hasil.

Streamlit ini **hanya untuk inference/demo**. Tahap rename dataset, konversi polygon ke bbox, split dataset, training YOLO, serta evaluasi Pipeline A/B dilakukan di notebook sebelumnya. Urutan notebook dijelaskan di [docs/notebook_workflow.md](docs/notebook_workflow.md).

## Struktur Folder

```text
alpr-lampung/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── pipeline_a.py
│   ├── pipeline_b.py
│   ├── ocr_utils.py
│   ├── region_mapper.py
│   ├── visualization.py
│   └── config.py
├── models/
│   └── yolov8n/
│       └── weights/
│           └── best.pt
├── results/
│   ├── pipeline_a/
│   │   └── technical_outputs/
│   └── pipeline_b/
│       └── technical_outputs/
├── assets/
│   ├── sample_images/
│   └── screenshots/
├── outputs/
│   ├── crops/
│   ├── roi/
│   ├── visualizations/
│   └── temp/
├── notebooks/
└── docs/
    └── notebook_workflow.md
```

Semua path runtime di aplikasi memakai path relatif repo melalui `src/config.py`, misalnya:

```text
models/yolov8n/weights/best.pt
results/pipeline_a
results/pipeline_b
assets/sample_images
outputs
```

Aplikasi tidak memakai path Google Drive seperti `/content/drive/MyDrive/...`.

## Instalasi

Disarankan memakai Python 3.12 dan virtual environment baru. `paddlepaddle==3.2.2` belum menyediakan wheel untuk Python 3.14, sehingga dependency ML/OCR di `requirements.txt` hanya akan diinstall otomatis pada Python `<3.13`.

```bash
cd alpr-lampung
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Setelah terbuka di browser:

1. Upload gambar kendaraan.
2. Pilih mode pipeline di sidebar.
3. Klik **Run Inference**.
4. Lihat hasil plat, wilayah, OCR mentah, visualisasi, dan detail teknis.

## Deploy ke Streamlit Community Cloud

PaddlePaddle perlu Python yang kompatibel. Jika log deploy menunjukkan:

```text
Using Python 3.14.5 environment
paddlepaddle==3.2.2 has no wheels with a matching Python ABI
```

maka ubah versi Python langsung dari dashboard Streamlit Community Cloud:

1. Buka app di Streamlit Community Cloud.
2. Masuk ke **Manage app**.
3. Buka **Settings** atau **Advanced settings**.
4. Pilih Python version **3.12**.
5. Simpan, lalu **Reboot** atau **Redeploy** app.

File `runtime.txt` dan `.python-version` tetap disediakan untuk dokumentasi/runtime lain, tetapi pada Streamlit Community Cloud versi Python perlu dipilih dari pengaturan app.

Jika Streamlit Cloud tetap menjalankan Python 3.14, aplikasi tetap bisa deploy dan terbuka, tetapi fitur inferensi Pipeline A/B akan menampilkan pesan bahwa PaddleOCR/YOLO belum tersedia. Untuk inferensi penuh, Python app harus diubah ke 3.12.

## Menaruh Model YOLO

File model final dari Google Drive:

```text
/content/drive/MyDrive/ALPR-Lampung-final/models/yolov8n/weights/best.pt
```

Salin ke repo:

```text
models/yolov8n/weights/best.pt
```

Jika `best.pt` terlalu besar untuk GitHub, jangan commit file tersebut. Simpan model di Google Drive/Hugging Face, lalu tulis instruksi agar user menaruhnya secara manual ke `models/yolov8n/weights/best.pt`.

Jika file `best.pt` belum ada, mode **Pipeline A** dan **Compare A vs B** tidak bisa dijalankan. Mode **Pipeline B** tetap bisa berjalan karena tidak memakai YOLO.

## Artefak Hasil Notebook

Folder `results/` menyimpan ringkasan evaluasi final dari notebook Pipeline A dan Pipeline B. File ini tidak wajib untuk inference satu gambar, tetapi berguna untuk dokumentasi dashboard, laporan, dan perbandingan eksperimen.

### Pipeline A

Salin dari:

```text
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_a/
```

Ke:

```text
results/pipeline_a/
```

File ramah pembaca yang disarankan:

```text
hasil_pipeline_a.csv
ringkasan_evaluasi_pipeline_a.csv
analisis_kesalahan_pipeline_a.csv
perbandingan_sebelum_sesudah.csv
dashboard_ringkas_pipeline_a.csv
```

Salin juga isi:

```text
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_a/technical_outputs/
```

Ke:

```text
results/pipeline_a/technical_outputs/
```

File teknis yang disarankan:

```text
ocr_results_improved_v2.csv
evaluation_summary_improved_v2.json
evaluation_summary_improved_v2.csv
candidate_debug_v2.csv
error_analysis_improved_v2.csv
comparison_improved_v1_vs_v2.csv
```

### Pipeline B

Salin dari:

```text
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_b/
```

Ke:

```text
results/pipeline_b/
```

File ramah pembaca yang disarankan:

```text
hasil_pipeline_b.csv
ringkasan_evaluasi_pipeline_b.csv
analisis_kesalahan_pipeline_b.csv
dashboard_ringkas_pipeline_b.csv
perbandingan_pipeline_a_vs_b.csv
```

Salin juga isi:

```text
/content/drive/MyDrive/ALPR-Lampung-results/pipeline_b/technical_outputs/
```

Ke:

```text
results/pipeline_b/technical_outputs/
```

File teknis yang disarankan:

```text
ocr_results_pipeline_b.csv
evaluation_summary_pipeline_b.json
candidate_debug_pipeline_b.csv
error_analysis_pipeline_b.csv
```

## Sample Image dan Screenshot

Untuk demo, salin beberapa gambar saja dari:

```text
/content/drive/MyDrive/ALPR-Lampung-final/test/images/
```

Ke:

```text
assets/sample_images/
```

Jangan menyalin seluruh dataset train/valid/test ke repo jika ingin repo tetap ringan.

Folder `assets/screenshots/` berisi screenshot dashboard atau visualisasi hasil dari notebook Pipeline A/B. Screenshot ini hanya untuk dokumentasi README/laporan, bukan untuk proses inference.

Jika repo akan dibuat public, pertimbangkan untuk menyamarkan sebagian plat pada sample image atau memakai gambar yang memang aman untuk demo.

## Fitur Dashboard

- Upload gambar kendaraan.
- Pipeline A: YOLOv8 mendeteksi plat, crop/ROI, PaddleOCR, candidate extraction, mapping wilayah.
- Pipeline B: PaddleOCR langsung dari full image, candidate extraction, mapping wilayah.
- Compare A vs B.
- Input manual plat untuk mengecek mapping wilayah.
- UI berbahasa Indonesia.
- Detail teknis ditempatkan di `Detail Teknis` agar tampilan utama tetap ramah pengguna.

## Catatan Privasi

Gambar plat kendaraan mengandung data sensitif. Hindari menyimpan, membagikan, atau mempublikasikan gambar kendaraan/plat tanpa izin. Folder `outputs/` masuk `.gitignore` agar file hasil upload/inference user tidak ikut ter-commit.
