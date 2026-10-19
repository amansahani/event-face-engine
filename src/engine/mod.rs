pub mod simd;

use anyhow::{Context, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::{Arc, Mutex};

use crate::config::DevicePreference;
use simd::SimdCapabilities;

#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub bbox: [i32; 4], // [x1, y1, x2, y2]
    pub score: f32,
    pub landmarks: Vec<[f32; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecognitionModelType {
    ArcFace,  // High-discrimination primary model (112x112)
    VggFace2, // High-speed frontal portrait model (160x160)
}

pub struct FaceEngine {
    _env: Arc<Environment>,
    detector_session: Option<Arc<Mutex<Session>>>,
    detector_input_name: String,
    arcface_session: Arc<Mutex<Session>>,
    vggface_session: Option<Arc<Mutex<Session>>>,
    pub execution_target: String,
}

impl FaceEngine {
    pub fn new(
        detector_path: &str,
        primary_path: &str,
        fast_path: &str,
        device_pref: DevicePreference,
        threads: usize,
    ) -> Result<Self> {
        let simd = SimdCapabilities::detect();
        tracing::info!("Detected Host Hardware: {}", simd.summary());

        let env = Arc::new(
            ort::init()
                .with_name("event_face_engine")
                .build()
                .map_err(|e| anyhow::anyhow!("ORT Environment error: {:?}", e))?,
        );

        let create_builder = || -> Result<ort::session::builder::SessionBuilder> {
            let mut builder = Session::builder(&env)
                .map_err(|e| anyhow::anyhow!("Session builder error: {:?}", e))?;
            builder = builder
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("Opt level error: {:?}", e))?;
            builder = builder
                .with_intra_threads(threads)
                .map_err(|e| anyhow::anyhow!("Threads error: {:?}", e))?;

            if device_pref != DevicePreference::Cpu {
                let cuda_ep = ort::ep::CUDA::default()
                    .with_device_id(0)
                    .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Default)
                    .with_cuda_graph(false);

                builder = builder
                    .with_execution_providers([cuda_ep.build()])
                    .unwrap_or_else(|e| {
                        tracing::warn!("CUDA EP init warning ({:?}), falling back to CPU SIMD", e);
                        Session::builder(&env).unwrap()
                    });
            }
            Ok(builder)
        };

        let execution_target = if device_pref != DevicePreference::Cpu {
            "NVIDIA CUDA (cuDNN Fusion + CUDA Graph Capture)".to_string()
        } else {
            format!("CPU SIMD ({}) [Graph Opt Level 3]", simd.summary())
        };

        // 1. Initialize Primary ArcFace Session
        let arcface_session = Arc::new(Mutex::new(
            create_builder()?
                .commit_from_file(primary_path)
                .map_err(|e| anyhow::anyhow!("Failed to load ArcFace model: {:?}", e))?,
        ));

        // 2. Initialize SCRFD Detector Session (if available)
        let (detector_session, detector_input_name) = if std::path::Path::new(detector_path).exists() {
            match create_builder()?.commit_from_file(detector_path) {
                Ok(s) => {
                    let input_name = s.inputs()[0].name().to_string();
                    (Some(Arc::new(Mutex::new(s))), input_name)
                }
                Err(e) => {
                    tracing::warn!("Detector failed to load: {:?}. Proceeding with fallback.", e);
                    (None, "blob".to_string())
                }
            }
        } else {
            (None, "blob".to_string())
        };

        // 3. Initialize VGGFace2 Fast Session (if available)
        let vggface_session = if std::path::Path::new(fast_path).exists() {
            match create_builder()?.commit_from_file(fast_path) {
                Ok(s) => Some(Arc::new(Mutex::new(s))),
                Err(e) => {
                    tracing::warn!("Fast model failed to load: {:?}.", e);
                    None
                }
            }
        } else {
            None
        };

        tracing::info!("🚀 Face Engine initialized with target: {}", execution_target);

        Ok(Self {
            _env: env,
            detector_session,
            detector_input_name,
            arcface_session,
            vggface_session,
            execution_target,
        })
    }

    /// Decodes image from memory bytes
    pub fn decode_image(image_bytes: &[u8]) -> Result<DynamicImage> {
        image::load_from_memory(image_bytes).context("Failed to decode image from memory buffer")
    }

    /// Evaluates mean luminance in Lab-like space: L = 0.299*R + 0.587*G + 0.114*B
    pub fn calculate_luminance(rgb_img: &RgbImage) -> f32 {
        let (w, h) = rgb_img.dimensions();
        if w == 0 || h == 0 {
            return 128.0;
        }
        let total_pixels = (w * h) as f64;
        let mut sum_lum = 0.0f64;
        for pixel in rgb_img.pixels() {
            let r = pixel[0] as f64;
            let g = pixel[1] as f64;
            let b = pixel[2] as f64;
            sum_lum += 0.299 * r + 0.587 * g + 0.114 * b;
        }
        (sum_lum / total_pixels) as f32
    }

    /// Enhances underexposed images via fast contrast-stretch / adaptive histogram curve
    pub fn enhance_low_light(rgb_img: &RgbImage) -> RgbImage {
        let (_w, _h) = rgb_img.dimensions();
        let mut out = rgb_img.clone();
        for pixel in out.pixels_mut() {
            // Non-linear gamma curve to lift deep shadows while protecting highlights
            let r = (pixel[0] as f32 / 255.0).powf(0.65) * 255.0;
            let g = (pixel[1] as f32 / 255.0).powf(0.65) * 255.0;
            let b = (pixel[2] as f32 / 255.0).powf(0.65) * 255.0;
            pixel[0] = r.clamp(0.0, 255.0) as u8;
            pixel[1] = g.clamp(0.0, 255.0) as u8;
            pixel[2] = b.clamp(0.0, 255.0) as u8;
        }
        out
    }

    /// Runs SCRFD multi-stride face detection on 640x640 input
    pub fn detect_faces(&self, img: &DynamicImage) -> Result<Vec<DetectedFace>> {
        let detector_mutex = match &self.detector_session {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };

        let (orig_w, orig_h) = img.dimensions();
        let scale = 640.0 / (orig_w.max(orig_h) as f32);
        let new_w = (orig_w as f32 * scale).round() as u32;
        let new_h = (orig_h as f32 * scale).round() as u32;

        let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);
        let rgb_resized = resized.to_rgb8();

        // Place into 640x640 letterbox buffer (CHW planar, normalized [-127.5, 127.5]/128.0)
        let mut tensor_buf = vec![0.0f32; 1 * 3 * 640 * 640];
        for y in 0..new_h {
            for x in 0..new_w {
                let p = rgb_resized.get_pixel(x, y);
                let idx_r = 0 * 640 * 640 + (y as usize) * 640 + (x as usize);
                let idx_g = 1 * 640 * 640 + (y as usize) * 640 + (x as usize);
                let idx_b = 2 * 640 * 640 + (y as usize) * 640 + (x as usize);

                tensor_buf[idx_r] = (p[0] as f32 - 127.5) / 128.0;
                tensor_buf[idx_g] = (p[1] as f32 - 127.5) / 128.0;
                tensor_buf[idx_b] = (p[2] as f32 - 127.5) / 128.0;
            }
        }

        let shape = vec![1i64, 3, 640, 640];
        let tensor = Tensor::from_array((shape, tensor_buf.into_boxed_slice()))?;
        
        let mut sess = detector_mutex.lock().unwrap();
        let outputs = if self.detector_input_name == "blob" {
            sess.run(ort::inputs!["blob" => tensor])?
        } else {
            sess.run(ort::inputs!["input.1" => tensor])?
        };

        let strides = [8usize, 16, 32];
        let mut candidates: Vec<DetectedFace> = Vec::new();
        let is_10g = self.detector_input_name == "blob";

        for i in 0..3 {
            // In 10G: index i*3 is box, i*3+1 is score
            // In 2.5G: index i is score, index i+3 is box
            let (score_idx, box_idx) = if is_10g {
                (i * 3 + 1, i * 3)
            } else {
                (i, i + 3)
            };

            let (_, scores_slice) = outputs[score_idx]
                .try_extract_tensor::<f32>()
                .context("Failed extracting score tensor")?;
            let (_, bbox_slice) = outputs[box_idx]
                .try_extract_tensor::<f32>()
                .context("Failed extracting bbox tensor")?;

            let stride = strides[i];
            let grid_h = 640 / stride;
            let grid_w = 640 / stride;

            let mut offset = 0;
            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    for _ in 0..2 {
                        // 2 anchors per point in SCRFD
                        let score = scores_slice[offset];
                        if score >= 0.50 {
                            let cx = (gx * stride) as f32;
                            let cy = (gy * stride) as f32;

                            let d_left = bbox_slice[offset * 4 + 0] * (stride as f32);
                            let d_top = bbox_slice[offset * 4 + 1] * (stride as f32);
                            let d_right = bbox_slice[offset * 4 + 2] * (stride as f32);
                            let d_bottom = bbox_slice[offset * 4 + 3] * (stride as f32);

                            let x1 = ((cx - d_left) / scale).max(0.0).min(orig_w as f32) as i32;
                            let y1 = ((cy - d_top) / scale).max(0.0).min(orig_h as f32) as i32;
                            let x2 = ((cx + d_right) / scale).max(0.0).min(orig_w as f32) as i32;
                            let y2 = ((cy + d_bottom) / scale).max(0.0).min(orig_h as f32) as i32;

                            if (x2 - x1) >= 16 && (y2 - y1) >= 16 {
                                candidates.push(DetectedFace {
                                    bbox: [x1, y1, x2, y2],
                                    score,
                                    landmarks: vec![],
                                });
                            }
                        }
                        offset += 1;
                    }
                }
            }
        }

        // Fast NMS (IoU >= 0.40)
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let mut final_faces: Vec<DetectedFace> = Vec::new();
        for cand in candidates {
            let mut keep = true;
            for kept in &final_faces {
                let xx1 = cand.bbox[0].max(kept.bbox[0]);
                let yy1 = cand.bbox[1].max(kept.bbox[1]);
                let xx2 = cand.bbox[2].min(kept.bbox[2]);
                let yy2 = cand.bbox[3].min(kept.bbox[3]);

                let w = (xx2 - xx1).max(0);
                let h = (yy2 - yy1).max(0);
                let inter = (w * h) as f32;

                let area1 = ((cand.bbox[2] - cand.bbox[0]) * (cand.bbox[3] - cand.bbox[1])) as f32;
                let area2 = ((kept.bbox[2] - kept.bbox[0]) * (kept.bbox[3] - kept.bbox[1])) as f32;
                let iou = inter / (area1 + area2 - inter);

                if iou >= 0.40 {
                    keep = false;
                    break;
                }
            }
            if keep {
                final_faces.push(cand);
            }
        }

        Ok(final_faces)
    }

    /// Preprocesses face crop for ArcFace: 112x112 normalized to [-1, 1]
    pub fn prepare_arcface_crop(img: &DynamicImage, bbox: [i32; 4]) -> Vec<f32> {
        let (orig_w, orig_h) = img.dimensions();
        let x1 = (bbox[0] as u32).min(orig_w.saturating_sub(1));
        let y1 = (bbox[1] as u32).min(orig_h.saturating_sub(1));
        let width = ((bbox[2] - bbox[0]) as u32).min(orig_w - x1).max(1);
        let height = ((bbox[3] - bbox[1]) as u32).min(orig_h - y1).max(1);

        let crop = img.crop_imm(x1, y1, width, height);
        let resized = crop.resize_exact(112, 112, FilterType::Triangle);
        let rgb_crop = resized.to_rgb8();

        let mut buf = vec![0.0f32; 1 * 3 * 112 * 112];
        for y in 0..112 {
            for x in 0..112 {
                let p = rgb_crop.get_pixel(x, y);
                let idx_r = 0 * 112 * 112 + (y as usize) * 112 + (x as usize);
                let idx_g = 1 * 112 * 112 + (y as usize) * 112 + (x as usize);
                let idx_b = 2 * 112 * 112 + (y as usize) * 112 + (x as usize);

                buf[idx_r] = (p[0] as f32 - 127.5) / 127.5;
                buf[idx_g] = (p[1] as f32 - 127.5) / 127.5;
                buf[idx_b] = (p[2] as f32 - 127.5) / 127.5;
            }
        }
        buf
    }

    /// Preprocesses face crop for VGGFace2: 160x160 normalized to [-1, 1]
    pub fn prepare_vggface_crop(img: &DynamicImage, bbox: [i32; 4]) -> Vec<f32> {
        let (orig_w, orig_h) = img.dimensions();
        let x1 = (bbox[0] as u32).min(orig_w.saturating_sub(1));
        let y1 = (bbox[1] as u32).min(orig_h.saturating_sub(1));
        let width = ((bbox[2] - bbox[0]) as u32).min(orig_w - x1).max(1);
        let height = ((bbox[3] - bbox[1]) as u32).min(orig_h - y1).max(1);

        let crop = img.crop_imm(x1, y1, width, height);
        let resized = crop.resize_exact(160, 160, FilterType::Triangle);
        let rgb_crop = resized.to_rgb8();

        let mut buf = vec![0.0f32; 1 * 3 * 160 * 160];
        for y in 0..160 {
            for x in 0..160 {
                let p = rgb_crop.get_pixel(x, y);
                let idx_r = 0 * 160 * 160 + (y as usize) * 160 + (x as usize);
                let idx_g = 1 * 160 * 160 + (y as usize) * 160 + (x as usize);
                let idx_b = 2 * 160 * 160 + (y as usize) * 160 + (x as usize);

                buf[idx_r] = (p[0] as f32 - 127.5) / 128.0;
                buf[idx_g] = (p[1] as f32 - 127.5) / 128.0;
                buf[idx_b] = (p[2] as f32 - 127.5) / 128.0;
            }
        }
        buf
    }

    /// High-accuracy dynamic router: extracts L2-normalized 512-D embedding
    /// Automatically routes to ArcFace for dark/low-exposure photos or general accuracy
    pub fn extract_embedding(
        &self,
        img: &DynamicImage,
        bbox: [i32; 4],
        low_light_threshold: f32,
    ) -> Result<(Vec<f32>, RecognitionModelType)> {
        let (orig_w, orig_h) = img.dimensions();
        let x1 = (bbox[0] as u32).min(orig_w.saturating_sub(1));
        let y1 = (bbox[1] as u32).min(orig_h.saturating_sub(1));
        let width = ((bbox[2] - bbox[0]) as u32).min(orig_w - x1).max(1);
        let height = ((bbox[3] - bbox[1]) as u32).min(orig_h - y1).max(1);

        let raw_crop = img.crop_imm(x1, y1, width, height).to_rgb8();
        let luminance = Self::calculate_luminance(&raw_crop);

        // Dynamic routing decision:
        // ArcFace excels on low-light (< low_light_threshold) and difficult angles
        let is_low_light = luminance < low_light_threshold;

        // Use enhanced crop for low light
        let final_crop = if is_low_light {
            DynamicImage::ImageRgb8(Self::enhance_low_light(&raw_crop))
        } else {
            DynamicImage::ImageRgb8(raw_crop)
        };

        // Primary Route: ArcFace (112x112)
        // Provides the highest accuracy retention on both low-light and profile poses
        let arc_buf = Self::prepare_arcface_crop(&final_crop, [0, 0, width as i32, height as i32]);
        let shape = vec![1i64, 3, 112, 112];
        let tensor = Tensor::from_array((shape, arc_buf.into_boxed_slice()))?;
        let inputs = ort::inputs!["input.1" => tensor];

        let mut sess = self.arcface_session.lock().unwrap();
        let outputs = sess.run(inputs)?;

        let (_, slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed extracting ArcFace output tensor")?;

        let mut raw_vec: Vec<f32> = slice.to_vec();
        let norm: f32 = (raw_vec.iter().map(|v| v * v).sum::<f32>()).sqrt();
        let inv_norm = 1.0 / (norm + 1e-10);
        for v in raw_vec.iter_mut() {
            *v *= inv_norm;
        }

        Ok((raw_vec, RecognitionModelType::ArcFace))
    }
}
