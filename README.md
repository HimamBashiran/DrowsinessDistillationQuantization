# Deteksi Kantuk Pengemudi Menggunakan Transformer dengan Teknik Kompresi Model Berbasis Distilasi dan Kuantisasi

Repositori ini berisi implementasi tugas akhir mengenai sistem **deteksi kantuk pengemudi berbasis sekuens temporal**. Untuk mengatasi keterbatasan komputasi pada perangkat *edge*, penelitian ini mengombinasikan arsitektur **Transformer** dengan teknik kompresi model melalui **Knowledge Distillation (KD)** dan **Quantization-Aware Training (QAT) INT8** berbasis *Straight-Through Estimator (STE)*.

Seluruh alur eksperimen, mulai dari ekstraksi fitur spasio-temporal, pemodelan baseline, pelatihan model **Teacher** dan **Student**, hingga pengujian akhir serta visualisasi atensi temporal, diimplementasikan dalam Jupyter Notebook `main_telup.ipynb`.

---

## 📌 Fitur & Metodologi

* **Temporal Transformer**
  Mengolah sekuens sebanyak **32 frame** dengan fitur berdimensi **1.029** untuk menangkap pola temporal yang berkaitan dengan kondisi kantuk.

* **Knowledge Distillation (KD)**
  Mentransfer pengetahuan dari model **Teacher** berkapasitas besar ke model **Student** yang lebih ringan melalui distilasi berbasis logit.

* **Quantization-Aware Training (QAT INT8)**
  Mensimulasikan proses kuantisasi presisi 8-bit selama proses training menggunakan *Straight-Through Estimator (STE)* untuk meningkatkan efisiensi inferensi.

* **Robustness terhadap Variasi Atribut Pengemudi**
  Menguji ketahanan model terhadap variasi atribut seperti **masker, kacamata putih, kacamata hitam, kondisi normal/polos, dan topi**.

* **Temporal Attention Explainability**
  Memvisualisasikan bobot atensi token `[CLS]` untuk menganalisis rentang waktu yang menjadi fokus model dalam mendeteksi kantuk.

---

## 🧠 Pipeline Penelitian

```text
Video Dataset
     │
     ▼
Feature Extraction
     │
     ├── Visual Features (DINOv2-Large)
     ├── Geometric Features (Face Landmarks)
     │
     ▼
1029-Dimensional Frame Features
     │
     ▼
Temporal Sequence (32 Frames)
     │
     ├───────────────┬────────────────┐
     ▼               ▼                ▼
Baseline Models   Transformer      Transformer
RNN/LSTM/CNN1D      Teacher           Student
                       │                │
                       │                │
                       └──────┬─────────┘
                              ▼
                    Knowledge Distillation
                              +
                    Quantization-Aware Training
                              │
                              ▼
                     INT8 Student Model
                              │
                              ▼
                  Final Testing & Benchmark
                              │
                              ▼
                  Temporal Attention Analysis
```

---

## 📊 Hasil Eksperimen & Benchmark

Evaluasi dilakukan secara independen pada **hold-out test set (Subjek 16–20)**.

| Model                              |   Accuracy |  Precision |     Recall |   Macro F1 | Latency (ms) |      FPS |    Masker | Kacamata Putih | Kacamata Hitam | Normal |       Topi |
| :--------------------------------- | ---------: | ---------: | ---------: | ---------: | -----------: | -------: | --------: | -------------: | -------------: | -----: | ---------: |
| **RNN**                            |     0.5250 |     0.5251 |     0.5250 |     0.5247 |         3.59 |    278.7 |     50.0% |          75.0% |          37.5% |  37.5% |      62.5% |
| **LSTM**                           |     0.5500 |     0.5549 |     0.5500 |     0.5396 |         8.27 |    120.9 |     50.0% |          75.0% |          50.0% |  50.0% |      50.0% |
| **CNN1D**                          |     0.8750 |     0.8836 |     0.8750 |     0.8743 |         4.28 |    233.5 |     62.5% |         100.0% |          87.5% | 100.0% |      87.5% |
| **Transformer Teacher (FP32)**     | **0.9250** | **0.9261** | **0.9250** | **0.9250** |       161.34 |      6.2 | **87.5%** |         100.0% |      **87.5%** | 100.0% |      87.5% |
| **Transformer Student (Baseline)** |     0.8750 |     0.8836 |     0.8750 |     0.8743 |        15.99 |     62.5 |     62.5% |         100.0% |          75.0% | 100.0% | **100.0%** |
| **Transformer Student (QAT + KD)** | **0.9250** | **0.9261** | **0.9250** | **0.9250** |    **10.47** | **95.5** | **87.5%** |         100.0% |          75.0% | 100.0% | **100.0%** |

> **Temuan Utama:** Model **Transformer Student dengan KD + QAT INT8** mampu mencapai performa yang setara dengan Teacher dengan **Macro F1 sebesar 0.9250**, sementara latency turun dari **161.34 ms menjadi 10.47 ms**. Hal ini menghasilkan throughput sebesar **95.5 FPS** dan peningkatan kecepatan sekitar **15.41×** dibandingkan Transformer Teacher.

---

## 🔬 Perbandingan Model

### Transformer Teacher vs. Student QAT + KD

| Metrik     | Teacher FP32 | Student QAT + KD | Perubahan |
| :--------- | -----------: | ---------------: | --------: |
| Accuracy   |       0.9250 |       **0.9250** |    Setara |
| Precision  |       0.9261 |       **0.9261** |    Setara |
| Recall     |       0.9250 |       **0.9250** |    Setara |
| Macro F1   |       0.9250 |       **0.9250** |    Setara |
| Latency    |    161.34 ms |     **10.47 ms** |  ↓ 93.51% |
| Throughput |      6.2 FPS |     **95.5 FPS** |  ↑ 15.41× |

Hasil tersebut menunjukkan bahwa kombinasi **Knowledge Distillation** dan **Quantization-Aware Training** dapat menghasilkan model Student yang jauh lebih efisien tanpa mengorbankan performa klasifikasi pada *hold-out test set*.

---

## 📁 Struktur Direktori

```text
.
├── drowsiness dataset/
│   ├── drowsiness/
│   │   ├── subject_01/
│   │   ├── subject_02/
│   │   └── ... (hingga subject_20)
│   │
│   ├── non-drowsiness/
│   │   ├── subject_01/
│   │   ├── subject_02/
│   │   └── ... (hingga subject_20)
│   │
│   └── hasil-extract-2/
│       ├── subject_01_feat.npy
│       ├── subject_02_feat.npy
│       └── ... (file ekstraksi fitur .npy)
│
├── result/
│   └── models/
│       ├── best_teacher_telup_final.pth
│       ├── best_student_baseline_telup_final.pth
│       └── transformer_student_qat_kd_final.pth
│
├── main_telup.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Alur Eksekusi

Seluruh eksperimen dilakukan melalui notebook:

```text
main_telup.ipynb
```

### 1. Feature Extraction

Melakukan ekstraksi fitur spasial dari setiap frame video pada folder:

* `drowsiness/`
* `non-drowsiness/`

Fitur hasil ekstraksi kemudian disimpan dalam format `.npy` pada folder:

```text
hasil-extract-2/
```

### 2. Data Splitting & Cross-Validation

Dataset dibagi berdasarkan **subject** untuk mencegah *data leakage*.

* **Training Set:** Subjek 01–15
* **Hold-out Test Set:** Subjek 16–20
* **Cross-Validation:** Stratified Group K-Fold

### 3. Pelatihan Baseline Models

Beberapa model digunakan sebagai pembanding:

* RNN
* LSTM
* CNN-1D

Model dievaluasi menggunakan metrik klasifikasi seperti **Accuracy, Precision, Recall, dan Macro F1**.

### 4. Pelatihan Transformer Teacher

Model **Transformer Teacher FP32** dilatih menggunakan kapasitas model yang lebih besar sebagai model referensi.

Checkpoint terbaik disimpan pada:

```text
result/models/best_teacher_telup_final.pth
```

### 5. Pelatihan Transformer Student Baseline

Model Student dengan arsitektur yang lebih ringan dilatih tanpa Knowledge Distillation maupun Quantization-Aware Training.

Checkpoint:

```text
result/models/best_student_baseline_telup_final.pth
```

### 6. Knowledge Distillation + QAT

Model Student kemudian dilatih menggunakan kombinasi:

* **Knowledge Distillation**
* **Quantization-Aware Training**
* **INT8 Fake Quantization**
* **Straight-Through Estimator (STE)**

Loss yang digunakan mengombinasikan **Cross-Entropy Loss** dan **KL-Divergence Loss** untuk mentransfer pengetahuan dari Teacher ke Student.

Checkpoint akhir:

```text
result/models/transformer_student_qat_kd_final.pth
```

### 7. Final Testing & Explainability

Model akhir diuji pada **Subjek 16–20** untuk mengevaluasi:

* Accuracy
* Precision
* Recall
* Macro F1
* Inference Latency
* FPS / Throughput
* Robustness terhadap variasi atribut pengemudi

Selain itu, dilakukan visualisasi **Temporal Attention Map** menggunakan bobot atensi token `[CLS]` untuk menganalisis fokus temporal model.

---

## ⚙️ Instalasi

Clone repositori:

```bash
git clone <repository-url>
cd <repository-folder>
```

Install seluruh dependensi:

```bash
pip install -r requirements.txt
```

Kemudian jalankan Jupyter Notebook:

```bash
jupyter notebook main_telup.ipynb
```

> **Catatan:** Pastikan struktur folder dataset sesuai dengan struktur direktori yang dijelaskan pada bagian sebelumnya.

---

## 🧪 Konfigurasi Utama

Beberapa konfigurasi eksperimen utama:

| Parameter            |            Nilai            |
| :------------------- | :-------------------------: |
| Sequence Length      |          32 frames          |
| Feature Dimension    |            1,029            |
| Dataset Subjects     |              20             |
| Training Subjects    |            01–15            |
| Test Subjects        |            16–20            |
| Cross-Validation     |   Stratified Group K-Fold   |
| Teacher Precision    |             FP32            |
| Student Quantization |             INT8            |
| Distillation         |    Knowledge Distillation   |
| QAT Method           | STE-based Fake Quantization |

---

## 📈 Kesimpulan

Eksperimen menunjukkan bahwa pendekatan **Transformer + Knowledge Distillation + Quantization-Aware Training** dapat mempertahankan performa klasifikasi sekaligus meningkatkan efisiensi inferensi secara signifikan.

Model **Transformer Student (QAT + KD)** mencapai:

* **Accuracy:** 92.50%
* **Precision:** 92.61%
* **Recall:** 92.50%
* **Macro F1:** 92.50%
* **Latency:** 10.47 ms/sample
* **Throughput:** 95.5 FPS

Dibandingkan Transformer Teacher FP32, model Student menghasilkan **penurunan latency sebesar 93.51%** dengan performa klasifikasi yang tetap setara pada *hold-out test set*.

Pendekatan ini menunjukkan potensi penerapan model Transformer yang telah dikompresi untuk sistem **deteksi kantuk pengemudi pada perangkat dengan sumber daya komputasi terbatas (*edge devices*)**.

---

## 👤 Author

**Himam Bashiran**

Data Science Study Program
Telkom University Purwokerto