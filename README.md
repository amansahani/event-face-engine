# Event Face Engine 🦀

High-performance, hardware-agnostic, and storage-decoupled face matching & event photo distribution engine written in **Rust**, powered by **Axum**, **ONNX Runtime (C API)**, and **SQLite Vector Search**.

---

## ⚡ Features

- **Blazing Fast End-to-End Latency**:
  - **Selfie Query**: Sub-millisecond similarity search (< 100 µs dot-product).
  - **Batch Ingestion**: Hardware-accelerated SIMD image decode with fast bilinear resize and normalized embedding extraction.
- **Hardware-Agnostic Silicon Support**:
  - **NVIDIA CUDA**: Fused cuDNN convolutions with **CUDA Graph capture** to eliminate kernel launch latency.
  - **x86_64 (Intel & AMD)**: Dedicated AVX-512, AVX2, and FMA3 SIMD vector instructions.
  - **ARM64**: Native ARM-NEON and SVE/SVE2 auto-vectorization (Apple Silicon M1-M4, AWS Graviton, Raspberry Pi 5).
  - **RISC-V**: RVV (Standard Vector Extension 'V') support.
- **Storage Agnostic**: Pluggable storage architecture supporting Local Filesystem, MinIO, Amazon S3, or Cloudflare R2.
- **Zero-Dependency Vector Store**: Embedded SQLite database with contiguous memory dot-product search. No external vector database containers required.
- **Built-in Mobile PWA**: Responsive photo portal for event attendees to scan a selfie and find their photos instantly.

---

## 🚀 Quick Start

### 1. Download Model Weights
Download the pre-exported ONNX models from the [Latest Release](https://github.com/amansahani/event-face-engine/releases):
```bash
mkdir -p models data/photos
# Place vggface2_facenet_fp32.onnx in models/
```

### 2. Run the Engine
```bash
# Optional environment overrides:
export MODEL_PATH="models/vggface2_facenet_fp32.onnx"
export DB_PATH="data/event_face.db"
export PHOTOS_DIR="data/photos"
export DEVICE="cuda" # or "cpu"

cargo run --release
```

Server starts on `http://localhost:8080`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Hardware engine profile & active SIMD status |
| `POST` | `/api/events/:event_id/upload` | Multipart photo upload and embedding extraction |
| `POST` | `/api/events/:event_id/selfie_search` | Multipart selfie search returning matched photo URLs |
| `GET` | `/` | Web Mobile Selfie Portal |

---

## 🧠 Model Attribution & Citations

The face recognition models used by this engine are based on the **InceptionResnetV1** architecture trained on the **VGGFace2** dataset:

- **Original Paper**: *VGGFace2: A dataset for recognising faces across pose and age* (Cao et al., FG 2018)
- **PyTorch Port**: [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- **Hugging Face / Open Model Hub Reference**: [Hugging Face Models: InceptionResnetV1 (VGGFace2)](https://huggingface.co/models?search=vggface2)
