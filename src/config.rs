use serde::Deserialize;
use std::env;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub server_host: String,
    pub server_port: u16,
    pub detector_model_path: String,
    pub primary_model_path: String,
    pub fast_model_path: String,
    pub db_path: String,
    pub photos_dir: String,
    pub storage_backend: String, // "local", "s3", "minio", "r2"
    pub s3_endpoint: Option<String>,
    pub s3_bucket: Option<String>,
    pub s3_access_key: Option<String>,
    pub s3_secret_key: Option<String>,
    pub prefer_device: DevicePreference,
    pub intra_threads: usize,
    pub similarity_threshold: f32,
    pub low_light_threshold: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DevicePreference {
    Auto,
    Cuda,
    Cpu,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server_host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            server_port: env::var("SERVER_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            detector_model_path: env::var("DETECTOR_MODEL_PATH")
                .unwrap_or_else(|_| "models/scrfd_2.5g_bnkps.onnx".to_string()),
            primary_model_path: env::var("PRIMARY_MODEL_PATH")
                .unwrap_or_else(|_| "models/arcface_w600k_r50.onnx".to_string()),
            fast_model_path: env::var("FAST_MODEL_PATH")
                .unwrap_or_else(|_| "models/vggface2_facenet_fp32.onnx".to_string()),
            db_path: env::var("DB_PATH")
                .unwrap_or_else(|_| "data/event_face.db".to_string()),
            photos_dir: env::var("PHOTOS_DIR")
                .unwrap_or_else(|_| "data/photos".to_string()),
            storage_backend: env::var("STORAGE_BACKEND")
                .unwrap_or_else(|_| "local".to_string()),
            s3_endpoint: env::var("S3_ENDPOINT").ok(),
            s3_bucket: env::var("S3_BUCKET").ok(),
            s3_access_key: env::var("S3_ACCESS_KEY").ok(),
            s3_secret_key: env::var("S3_SECRET_KEY").ok(),
            prefer_device: match env::var("DEVICE").unwrap_or_default().to_lowercase().as_str() {
                "cuda" | "gpu" => DevicePreference::Cuda,
                "cpu" => DevicePreference::Cpu,
                _ => DevicePreference::Auto,
            },
            intra_threads: env::var("CPU_THREADS")
                .ok()
                .and_then(|t| t.parse().ok())
                .unwrap_or_else(|| {
                    let total = num_cpus::get();
                    if total > 2 { total / 2 } else { total }
                }),
            similarity_threshold: env::var("SIMILARITY_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.60),
            low_light_threshold: env::var("LOW_LIGHT_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(80.0),
        }
    }
}
