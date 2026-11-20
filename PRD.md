# PRD: Comparative Study of Landmark vs. Pixel Input for Driver Drowsiness Detection (Phase 1: Telup Dataset)

## 1. Tujuan & Ruang Lingkup
Penelitian ini membandingkan dua paradigma representasi input untuk klasifikasi kantuk (*non-drowsiness* = 0, *drowsiness* = 1) berbasis video pada **120 frame**:
- **Track A — Input Titik (Landmark & Geometric Sequence):** Sekuens fitur wajah ringan (`120 × 109`) menggunakan `RNN`, `CNN`, `LSTM`, dan `TRANSFORMER` skenario S4 (`TEACHER`, `STUDENT`, `KD`, `KD+QAT`).
- **Track B — Input Pixel (Spatiotemporal Video):** Klip wajah RGB ter-crop (`3 × 120 × 112 × 112`) menggunakan 4 arsitektur 3D CNN (`3D ResNet`, `3D WideResNet`, `3D ResNeXt`, `3D DenseNet`) yang di-porting bersih dari repositori `video-classification-3d-cnn-pytorch-master`.

**Fase 1:** Menyelesaikan seluruh eksperimen, komparasi metrik klasifikasi, efisiensi komputasi (**Parameters & GFLOPs**), dan **XAI (SHAP vs. Top-K 3D Score-CAM)** pada **Dataset Telup** di lingkungan **CPU lokal**.  
**Fase 2 (Future Scope):** Arsitektur kode dibuat modular agar dataset **NTHU-DDD** dapat langsung dipasang (*plug-and-play*) untuk studi komparasi lintas dataset.

---

## 2. Dataset & Protokol Evaluasi (Subject-Independent)
Agar perbandingan Track A dan Track B adil (*apple-to-apple*) dan bebas *data leakage*:
- **Dataset Telup:** 160 video dari 20 subjek (`drowsiness/` dan `non-drowsiness/`).
  - **Training Pool:** Subjek `01–15` (120 video) → **5-Fold `GroupKFold`** berbasis `subject` (masing-masing fold: 96 train, 24 val).
  - **Hold-out Test Set:** Subjek `16–20` (40 video) → Dievaluasi menggunakan **Soft-Voting Ensemble** dari 5 checkpoint fold.
- **Anti-Overfitting & Training Engine:**
  - Optimizer `AdamW`, `LinearLR` warmup + `CosineAnnealingLR`, `Label Smoothing`, `Gradient Clipping`, dan `Early Stopping` berbasis `val_loss`.
  - Seluruh scaler normalisasi dihitung **per-fold hanya dari data training fold aktif** dan disimpan ke dalam checkpoint.

---

## 3. Spesifikasi Track A — Input Titik (`120 × 109`)

### A. Preprocessing Track A
1. **Uniform Temporal Sampling:** Mengambil tepat `120` frame berjarak merata (`np.linspace`) per video.
2. **Photometric Normalization (CLAHE):** RGB → LAB, terapkan CLAHE pada channel `L` (`clipLimit=2.0`, `tileGridSize=(8, 8)`), lalu kembalikan ke RGB.
3. **Landmark Detection:** MediaPipe `FaceLandmarker` (`num_faces=1`); frame gagal deteksi diisi dengan *forward-fill* dari frame valid terakhir.
4. **Feature Engineering (`109` fitur/frame):**
   - `5` fitur skalar: `EAR` (rata-rata mata kiri & kanan), `MAR`, `Pitch`, `Yaw`, `Roll` (`cv2.solvePnP` + `RQDecomp3x3`).
   - `104` fitur koordinat `(x, y)` dari `52` landmark (mata kiri, mata kanan, mulut) yang di-*center* dan dinormalisasi terhadap jarak maksimumnya.
5. **Per-Fold Z-Score Standardization** (dihitung per-fold dari data train).
6. **Training-Only Temporal Augmentation:** Random channel scaling (`±10%`), Gaussian noise (`std=0.08`), dan random frame dropout (`10%`).

### B. Model Track A (7 Model)
1. `RNN` (Bidirectional RNN + LayerNorm + Mean Pooling)
2. `CNN` (1D-CNN + BatchNorm + AdaptiveAvgPool)
3. `LSTM` (Bidirectional LSTM + LayerNorm + Mean Pooling)
4. `TRANSFORMER TEACHER` — S4 (`hidden=32, layers=3, heads=2, ffn=128, cls=32`)
5. `TRANSFORMER STUDENT` — S4 tanpa KD (`hidden=16, layers=2, heads=2, ffn=64, cls=16`)
6. `TRANSFORMER KD` — Student S4 dengan Knowledge Distillation dari `TRANSFORMER TEACHER` (`T=3.0, alpha=0.5`)
7. `TRANSFORMER KD+QAT` — Student S4 dengan KD + Quantization-Aware Training INT8 (`FakeQuantizeSTE` & `QuantizedLinear`)

### C. XAI Track A — SHAP Value
- Menggunakan library `shap` pada model terbaik Track A.
- Menghasilkan:
  1. **Global Feature Importance** (`EAR`, `MAR`, `Pitch`, `Yaw`, `Roll`, serta grup landmark mata kiri, mata kanan, dan mulut).
  2. **Temporal Importance Profile** (kontribusi SHAP sepanjang frame `1..120` disandingkan dengan sinyal `EAR` & `MAR`).

---

## 4. Spesifikasi Track B — Input Pixel (`3 × 120 × 112 × 112`)

### A. Preprocessing Track B
1. **Uniform Temporal Sampling:** Menggunakan **120 indeks frame yang sama persis** dengan Track A.
2. **Face ROI Cropping:** Mendeteksi bounding box wajah tiap frame menggunakan MediaPipe (dengan margin proporsional; fallback ke bounding box valid terakhir jika gagal deteksi), lalu crop dan resize ke `112 × 112`.
3. **Caching ke Disk:** Klip wajah ter-crop disimpan di `hasil-extract-120-frame/telup/` (`.npy`) agar CPU lokal tidak mengulang ekstraksi video.
4. **Normalization & Augmentation:** Mean subtraction sesuai statistik Kinetics (`[114.7748, 107.7354, 99.4750]` pada skala 0–255), konversi dimensi ke `(C, T, H, W)`, serta augmentasi *training-only* yang konsisten sepanjang 120 frame (horizontal flip & jitter ringan).

### B. Model Track B (4 Model 3D CNN)
Di-porting bersih tanpa `nn.DataParallel` dan `opts.py` ke `src/models_pixel/`:
1. `3D ResNet` (`resnet-18`)
2. `3D WideResNet` (`wideresnet-50`, `k=2`)
3. `3D ResNeXt` (`resnext-50`, `cardinality=32`)
4. `3D DenseNet` (`densenet-121`)

**Strategi Training (CPU-Friendly & Fair):**
- Menggunakan **Pretrained Kinetics-400 + Partial Fine-Tuning** (layer awal dibekukan, **blok konvolusi terakhir + layer `fc` dilatih langsung** pada input pixel `120 × 112 × 112`).
- Disediakan 1 flag di `src/config.py` (`PIXEL_FINETUNE_MODE = "last_block"` atau `"full"`) agar tetap mudah diubah bila dijalankan di GPU Colab.

### C. XAI Track B — Top-K 3D Score-CAM
- Implementasi **3D Score-CAM** (*gradient-free*) pada layer konvolusi 3D terakhir sebelum Global Average Pooling, menggunakan seleksi **Top-K channel** paling aktif agar realistis dijalankan di CPU lokal.
- Menghasilkan overlay heatmap spasial-temporal di atas frame wajah asli untuk sampel *Drowsy* dan *Non-Drowsy*.

---

## 5. Metrik Evaluasi, Efisiensi & Aturan Visualisasi
- **Metrik Klasifikasi:** Accuracy, Precision, Recall, F1-Score (`mean ± std` untuk 5-Fold CV + nilai pada Hold-out Test) & Confusion Matrix.
- **Metrik Efisiensi (`src/benchmark.py`):**
  - `Total_Params` & `Trainable_Params`
  - `GFLOPs` per 1 forward pass (diukur menggunakan profiler seperti `thop` / `fvcore`)
  - `Size_KB` & `Latency_ms` (CPU warm-up + `perf_counter`)
- **Aturan Visualisasi (Bahasa Inggris):**
  - Seluruh judul grafik, label sumbu, legenda, dan anotasi menggunakan **Bahasa Inggris** standar publikasi ilmiah.
  - Menggunakan Matplotlib OO API (`layout="constrained"`), palet warna aksesibel (*Okabe-Ito*), dan pelaporan ketidakpastian yang jujur (`mean ± SD`).

---

## 6. Eksekusi Bertahap (Single-Model Execution)
Agar aman dijalankan di CPU lokal, pipeline mendukung **menjalankan 1 model terlebih dahulu** (mis. lewat `run_single_model("RNN")` atau `run_single_model("3D ResNet")`):
- Setiap model yang selesai otomatis menyimpan checkpoint ke `results/telup/checkpoints/` dan metriknya ke `results/telup/tables/per_model/`.
- Sel komparasi akhir otomatis menggabungkan seluruh hasil model yang sudah tersimpan di disk tanpa perlu mengulang training model lain.

---

## 7. Struktur Folder & File

```text
tugas akhir/
├── archive/                               # Arsip lama (tidak di-import)
│   ├── main_telup_120_4skenario...ipynb   # Notebook lama
│   └── video-classification-3d-cnn.../    # Repo asli 3D CNN
│
├── drowsiness dataset/                    # Dataset video asli Telup
│   ├── drowsiness/                        # Video kelas kantuk (1)
│   ├── non-drowsiness/                    # Video kelas normal (0)
│   └── hasil-extract-120-gabungan-pkl/    # Fitur titik .pkl (120x109)
│
├── pretrained_models/                     # Bobot model pihak ketiga
│   ├── face_landmarker.task               # Model landmark MediaPipe
│   └── kinetics_3dcnn/                    # Bobot pretrained Kinetics-400
│       ├── resnet-18-kinetics.pth         # Bobot 3D ResNet-18
│       ├── wideresnet-50-kinetics.pth     # Bobot 3D WideResNet-50
│       ├── resnext-50-kinetics.pth        # Bobot 3D ResNeXt-50
│       └── densenet-121-kinetics.pth      # Bobot 3D DenseNet-121
│
├── hasil-extract-120-frame/               # Cache klip wajah Track B (120 frame)
│   └── telup/                             # Klip wajah crop 120x112x112 (.npy)
│
├── src/                                   # Kode inti Python (modular)
│   ├── __init__.py                        # Penanda package Python
│   ├── config.py                          # Path, seed, hyperparameter
│   ├── preprocessing.py                   # Ekstraksi titik & face-crop
│   ├── datasets.py                        # Loader & Z-score per-fold
│   ├── models_point.py                    # RNN, CNN, LSTM, Transformer S4
│   ├── models_pixel/                      # 4 model 3D CNN bersih
│   │   ├── __init__.py                    # Pemanggil model + Partial Fine-Tune
│   │   ├── resnet.py                      # Arsitektur 3D ResNet
│   │   ├── wide_resnet.py                 # Arsitektur 3D WideResNet
│   │   ├── resnext.py                     # Arsitektur 3D ResNeXt
│   │   └── densenet.py                    # Arsitektur 3D DenseNet
│   ├── trainer.py                         # Training, 5-Fold CV, KD, Early Stop
│   ├── evaluator.py                       # Hold-out Test, simpan hasil per-model
│   ├── benchmark.py                       # Hitung Params, GFLOPs, Latency
│   └── xai.py                             # SHAP (titik) & Score-CAM (pixel)
│
├── tests/                                 # Tes cepat sebelum training
│   └── test_smoke.py                      # Cek shape, scaler, QAT, XAI
│
├── results/                               # Semua output hasil eksperimen
│   └── telup/                             # (Nanti tambah folder nthu_ddd/)
│       ├── checkpoints/                   # Bobot terbaik (.pth) per fold
│       │   ├── track_a/                   # Checkpoint model input titik
│       │   └── track_b/                   # Checkpoint model input pixel
│       ├── tables/                        # Semua tabel angka (.csv / .json)
│       │   ├── per_model/                 # Hasil terpisah tiap model yang selesai
│       │   ├── cv_fold_details.csv        # Gabungan skor detail Fold 1 s.d. 5
│       │   ├── summary_comparison.csv     # Tabel master F1 vs GFLOPs
│       │   └── shap_feature_importance.csv# Ranking nilai SHAP fitur titik
│       └── figures/                       # Semua grafik (.png) Bahasa Inggris
│           ├── eda/                       # Grafik distribusi & face-crop
│           ├── training_curves/           # Kurva loss/acc & confusion matrix
│           ├── comparison/                # Grafik F1 & GFLOPs semua model
│           ├── xai_shap/                  # Grafik SHAP (fitur & temporal)
│           └── xai_scorecam/              # Overlay Score-CAM pada wajah
│
├── PRD.md                                 # Dokumen acuan utama proyek
└── main_telup_point_vs_pixel.ipynb        # Notebook utama (bisa run 1 model dulu)
```

---

## 8. Prinsip Rekayasa Kode
- **Ponytail:** Kode minimal, tanpa duplikasi/bloat, tidak ada abstraksi berlebihan, wajib ada smoke check (`tests/test_smoke.py`), dan gunakan komentar `shortcut: <limit>, <upgrade>` jika ada kompromi sementara.
- **Scientific Code:** Bebas *data leakage*, replikasi di level subjek, pelaporan metrik beserta SD, dan grafik publikasi yang jujur serta aksesibel.
