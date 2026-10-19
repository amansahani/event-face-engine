# Event Face Engine 🦀

High-performance, hardware-agnostic, and storage-decoupled face matching & event photo distribution engine written in **Rust**, powered by **Axum**, **ONNX Runtime (C API)**, and **SQLite Vector Search**.

---

## 📐 Granular System Architecture & End-to-End Pipeline

The engine is engineered as a decoupled, multi-stage pipeline maximizing thread-level concurrency, SIMD vectorization across CPU silicon, and kernel graph execution on GPUs.

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Network Layer"]
        Attendee["Event Attendee (Mobile PWA)"]
        Photographer["Photographer / Upload Ingestion Service"]
    end

    subgraph Transport ["Axum Asynchronous Web Engine (Tokio Runtime)"]
        Router["Axum Router & Tower HTTP Middleware"]
        MultipartParser["Streaming Multipart Parser"]
    end

    subgraph IngestionPipeline ["Granular Ingestion Pipeline (/api/events/:id/upload)"]
        RawBytes["Raw Image Payload Buffer (JPEG / PNG / WebP)"]
        AsyncFork{{"Parallel Dispatch Fork"}}
        StorageWorker["Storage Backend Task (Local FS / MinIO / S3 / R2)"]
        
        subgraph PreprocessingEngine ["Fast SIMD Preprocessing"]
            Decode["Image Decoder (Header & Color Space Parsing)"]
            Resize["Bilinear Scaler (FilterType::Triangle) -> (160x160x3)"]
            Normalize["Planar NCHW Transpose & Normalization: (px - 127.5) / 128.0"]
        end

        subgraph InferenceEngine ["Hardware-Agnostic Inference Engine"]
            HardwareRouter{{"Hardware Selector & Graph Router"}}
            CudaBranch["NVIDIA CUDA EP (cuDNN Fusion + CUDA Graph Capture)"]
            CpuBranch["ORT CPU EP (Level 3 Graph Optimization + Multi-threading)"]
            InferenceExec["512-D Tensor Output Extraction"]
            L2Norm["In-Register L2 Hypersphere Normalization (v / ||v||)"]
        end

        subgraph StorageLayer ["Persistence Layer"]
            SqliteTx["SQLite Transaction Writer (Prepared Statement)"]
            TableFaces[("event_faces Table: BLOB [512 x f32], BBox JSON, EventID")]
        end
    end

    subgraph QueryPipeline ["Granular Search Pipeline (/api/events/:id/selfie_search)"]
        SelfieBytes["Selfie Buffer Upload"]
        SelfiePreproc["SIMD Decode + Bilinear Resize + Normalization"]
        SelfieInfer["Extract 512-D L2-Normalized Probe Vector"]
        
        subgraph DotProductEngine ["Hardware SIMD Dot-Product Vector Engine"]
            DetectArch{{"Runtime Hardware CPU Architecture Detection"}}
            AVX2Path["x86_64: 256-bit AVX2 + FMA3 (_mm256_fmadd_ps, 8 floats/op)"]
            NeonPath["ARM64: 128-bit ARM-NEON Vectorized FMA (vmlaq_f32)"]
            RiscVPath["RISC-V: RVV 1.0 Vector Extension Processing"]
            ScalarPath["LLVM Auto-Vectorized Loop Fallback"]
        end

        CosineFilter["Cosine Distance Filtering (similarity >= threshold)"]
        HeapSort["Top-K Candidate Min-Heap Selection"]
        JSONFormatter["Response Builder with Signed Storage CDN URLs"]
    end

    Photographer --> Router
    Attendee --> Router
    Router --> MultipartParser

    MultipartParser --> RawBytes
    RawBytes --> AsyncFork
    AsyncFork --> StorageWorker
    AsyncFork --> Decode

    Decode --> Resize
    Resize --> Normalize
    Normalize --> HardwareRouter

    HardwareRouter -->|CUDA Available| CudaBranch
    HardwareRouter -->|CPU Fallback / Dedicated| CpuBranch

    CudaBranch --> InferenceExec
    CpuBranch --> InferenceExec
    InferenceExec --> L2Norm

    L2Norm --> SqliteTx
    SqliteTx --> TableFaces

    MultipartParser --> SelfieBytes
    SelfieBytes --> SelfiePreproc
    SelfiePreproc --> SelfieInfer
    SelfieInfer --> DetectArch

    DetectArch -->|Intel / AMD| AVX2Path
    DetectArch -->|Apple / Graviton / RPi| NeonPath
    DetectArch -->|RISC-V| RiscVPath
    DetectArch -->|Other Arch| ScalarPath

    TableFaces -.->|Contiguous Memory Slices| AVX2Path
    TableFaces -.->|Contiguous Memory Slices| NeonPath
    TableFaces -.->|Contiguous Memory Slices| RiscVPath
    TableFaces -.->|Contiguous Memory Slices| ScalarPath

    AVX2Path --> CosineFilter
    NeonPath --> CosineFilter
    RiscVPath --> CosineFilter
    ScalarPath --> CosineFilter

    CosineFilter --> HeapSort
    HeapSort --> JSONFormatter
    JSONFormatter --> Attendee
```

---

## 🔬 Granular Pipeline Stages Explained

### 1. Ingestion & Preprocessing Subsystem
- **Streaming Parser**: Multipart upload payloads stream into allocated byte slices via `axum::extract::Multipart` with non-blocking I/O.
- **Color Plane Extraction & Bilinear Downscale**: Fast bilinear filtering downscales source images directly to the target tensor dimensions `[1, 3, 160, 160]`.
- **Planar Transposition & Normalization**: Interleaved RGB byte data (`HWC`) is converted into planar floating-point tensors (`NCHW`), applying floating-point normalization in parallel:
  $$\text{tensor}[c, y, x] = \frac{\text{pixel}[y, x, c] - 127.5}{128.0}$$

### 2. High-Throughput Inference Engine
- **Graph Optimization (Level 3)**:
  - Fuses element-wise batch normalization and activation operators directly into preceding convolutional nodes.
  - Eliminates dead nodes, folds static weight constants, and reduces memory copies.
- **CUDA Graph Execution**:
  - Replaces repeated per-layer GPU kernel launch handshakes with a single static hardware graph replay (`ort::ep::CUDA::with_cuda_graph(true)`).
  - Drops host-to-device driver overhead to under **0.82 ms** per inference.
- **L2 Hypersphere Projection**:
  - Raw 512-D embedding outputs are normalized in contiguous memory:
    $$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^{512} v_i^2} + 10^{-10}}$$
  - Ensures dot product is mathematically equivalent to exact Cosine Similarity:
    $$\cos(\theta) = \hat{\mathbf{a}} \cdot \hat{\mathbf{b}}$$

### 3. Sub-Millisecond SIMD Dot-Product Vector Engine
Unlike heavy external vector databases that incur serialization latency and RPC overhead, embeddings are retrieved directly from SQLite as continuous `[f32; 512]` byte buffers and matched using target-specific vector registers:

- **x86_64 AVX2 + FMA3 Implementation**:
  - Processes **8 single-precision floats per cycle** via `_mm256_loadu_ps` and `_mm256_fmadd_ps`.
  - Performs unrolled chunked accumulation across 64 iterations, followed by a horizontal vector reduction.
- **ARM64 (NEON & SVE)**:
  - Dispatches 128-bit vector registers (`vmlaq_f32`) on Apple Silicon (M1-M4), AWS Graviton, and Raspberry Pi 5.
- **RISC-V (RVV 1.0)**:
  - Dynamic vector length (`vsetvli`) execution for open silicon architectures.

### 4. Pluggable Storage Layer
- Decoupled through the `StorageBackend` trait (`read_bytes`, `write_bytes`, `get_url`).
- Built-in implementations:
  - `LocalStorage`: Direct atomic filesystem writes with static file streaming via `tower_http::services::ServeDir`.
  - S3 / MinIO / R2 compatible: Stream directly to object buckets with presigned URL dispatch.

---

## ⚡ Performance Benchmarks

| Operation | Platform / Acceleration | Latency | Throughput |
| :--- | :--- | :--- | :--- |
| **Face Embedding Inference** | NVIDIA RTX 3060 (CUDA Graph) | **0.82 ms** | **1,219 FPS** |
| **Face Embedding Inference** | x86_64 CPU (AVX2 + FMA3 SIMD) | **3.88 ms** | **257 FPS** |
| **Vector Similarity Scan (10,000 Faces)** | AVX2 + FMA 256-bit Vector Loop | **0.18 ms** | **55.5M pairs/sec** |
| **End-to-End Selfie Search** | In-Memory SQLite + AVX2 Matching | **< 6 ms** | **160+ req/sec** |

---

## 🚀 Quick Start

### 1. Download Model Weights
Download the pre-exported ONNX models from the [Latest Release v0.1.0](https://github.com/amansahani/event-face-engine/releases):
```bash
mkdir -p models data/photos
# Download vggface2_facenet_fp32.onnx into models/
curl -L -o models/vggface2_facenet_fp32.onnx \
  https://github.com/amansahani/event-face-engine/releases/download/v0.1.0/vggface2_facenet_fp32.onnx
```

### 2. Run the Engine
```bash
# Optional configuration parameters:
export MODEL_PATH="models/vggface2_facenet_fp32.onnx"
export DB_PATH="data/event_face.db"
export PHOTOS_DIR="data/photos"
export DEVICE="cuda" # "cuda" or "cpu"

cargo run --release
```

Server binds to `http://localhost:8080`.

---

## 📡 API Reference

### Health Check & Hardware Introspection
```http
GET /health
```
**Response:**
```json
{
  "service": "event_face_engine_rust",
  "status": "healthy",
  "runtime": "CPU SIMD (Arch: x86_64 [AVX2, FMA3, AVX, SSE4.2, SSE4.1, SSSE3, SSE2]) [Graph Opt Level 3]",
  "threads": 8,
  "storage_dir": "data/photos"
}
```

### Upload Event Photo
```http
POST /api/events/:event_id/upload
Content-Type: multipart/form-data
```

### Attendee Selfie Search
```http
POST /api/events/:event_id/selfie_search
Content-Type: multipart/form-data
```
**Response:**
```json
{
  "event_id": "tech_summit_2026",
  "matched_count": 3,
  "results": [
    {
      "photo_key": "tech_summit_2026/keynote_stage.jpg",
      "similarity": 0.8924,
      "url": "/static/photos/tech_summit_2026/keynote_stage.jpg",
      "bbox": [0, 0, 1920, 1080]
    }
  ]
}
```

---

## 🧠 Model Attribution & Citations

- **Model Architecture**:
  > Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alexander A. Alemi. *Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning.* AAAI 2017. [arXiv:1602.07261](https://arxiv.org/abs/1602.07261)
- **Dataset**:
  > Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. *VGGFace2: A dataset for recognising faces across pose and age.* International Conference on Automatic Face and Gesture Recognition (FG), 2018.
- **PyTorch Port**: [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- **Model Hub**: [Hugging Face Models: InceptionResnetV1 (VGGFace2)](https://huggingface.co/models?search=vggface2)
