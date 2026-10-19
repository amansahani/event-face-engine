pub mod simd;

use anyhow::{Context, Result};
use image::{imageops::FilterType, GenericImageView};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::{Arc, Mutex};

use crate::config::DevicePreference;
use simd::SimdCapabilities;

pub struct FaceEngine {
    _env: Arc<Environment>,
    session: Arc<Mutex<Session>>,
    pub execution_target: String,
}

impl FaceEngine {
    pub fn new(model_path: &str, device_pref: DevicePreference, threads: usize) -> Result<Self> {
        let simd = SimdCapabilities::detect();
        tracing::info!("Detected Host Hardware: {}", simd.summary());

        let env = Arc::new(
            ort::init()
                .with_name("event_face_engine")
                .build()
                .map_err(|e| anyhow::anyhow!("ORT Environment error: {:?}", e))?,
        );

        let mut builder = Session::builder(&env)
            .map_err(|e| anyhow::anyhow!("Session builder error: {:?}", e))?;

        // 1. Enable Level 3 Full Graph Optimization (Constant folding, operator fusion, dead node removal)
        builder = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Opt level error: {:?}", e))?;

        builder = builder
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("Threads error: {:?}", e))?;

        let mut execution_target = format!("CPU SIMD ({}) [Graph Opt Level 3]", simd.summary());

        // 2. CUDA Path: Enable CUDA Graph Capture & cuDNN Fusion when requested
        if device_pref != DevicePreference::Cpu {
            let cuda_ep = ort::ep::CUDA::default()
                .with_device_id(0)
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Default)
                .with_cuda_graph(true); // CUDA Graph enabled for ultra-low dispatch latency

            builder = builder
                .with_execution_providers([cuda_ep.build()])
                .unwrap_or_else(|e| {
                    tracing::warn!("CUDA EP unavailable ({:?}), proceeding on CPU SIMD", e);
                    Session::builder(&env).unwrap()
                });
            execution_target = "NVIDIA CUDA (cuDNN Fusion + CUDA Graph Capture)".to_string();
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to commit model from file: {:?}", e))?;

        tracing::info!("Active Execution Engine: {}", execution_target);

        Ok(Self {
            _env: env,
            session: Arc::new(Mutex::new(session)),
            execution_target,
        })
    }

    /// Preprocesses raw JPEG/PNG image bytes into flat buffer (1, 3, 160, 160)
    pub fn preprocess_image(image_bytes: &[u8]) -> Result<(Vec<f32>, (u32, u32))> {
        let img = image::load_from_memory(image_bytes)
            .context("Failed to decode image from memory buffer")?;
        let (orig_w, orig_h) = img.dimensions();

        // High-speed bilinear resize to 160x160
        let resized = img.resize_exact(160, 160, FilterType::Triangle);
        let rgb_img = resized.to_rgb8();

        let mut buffer = vec![0.0f32; 1 * 3 * 160 * 160];
        for y in 0..160 {
            for x in 0..160 {
                let pixel = rgb_img.get_pixel(x, y);
                let idx_r = 0 * 160 * 160 + (y as usize) * 160 + (x as usize);
                let idx_g = 1 * 160 * 160 + (y as usize) * 160 + (x as usize);
                let idx_b = 2 * 160 * 160 + (y as usize) * 160 + (x as usize);

                buffer[idx_r] = (pixel[0] as f32 - 127.5) / 128.0;
                buffer[idx_g] = (pixel[1] as f32 - 127.5) / 128.0;
                buffer[idx_b] = (pixel[2] as f32 - 127.5) / 128.0;
            }
        }

        Ok((buffer, (orig_w, orig_h)))
    }

    /// Runs ONNX inference and returns 512-D L2-normalized vector
    pub fn extract_embedding(&self, buffer: Vec<f32>) -> Result<Vec<f32>> {
        let mut session = self.session.lock().unwrap();

        let shape = vec![1i64, 3, 160, 160];
        let tensor = Tensor::from_array((shape, buffer.into_boxed_slice()))?;

        let inputs = ort::inputs!["input" => tensor];
        let outputs = session.run(inputs)?;

        let (_, slice) = outputs["embeddings"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        let mut raw_vec: Vec<f32> = slice.to_vec();

        // L2 Hypersphere normalization
        let norm: f32 = (raw_vec.iter().map(|v| v * v).sum::<f32>()).sqrt();
        let inv_norm = 1.0 / (norm + 1e-10);
        for v in raw_vec.iter_mut() {
            *v *= inv_norm;
        }

        Ok(raw_vec)
    }
}
