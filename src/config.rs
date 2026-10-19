use serde::Deserialize;
use std::env;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub server_host: String,
    pub server_port: u16,
    pub model_path: String,
    pub db_path: String,
    pub photos_dir: String,
    pub prefer_device: DevicePreference,
    pub intra_threads: usize,
    pub similarity_threshold: f32,
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
            model_path: env::var("MODEL_PATH")
                .unwrap_or_else(|_| "models/vggface2_facenet_fp32.onnx".to_string()),
            db_path: env::var("DB_PATH")
                .unwrap_or_else(|_| "data/event_face.db".to_string()),
            photos_dir: env::var("PHOTOS_DIR")
                .unwrap_or_else(|_| "data/photos".to_string()),
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
        }
    }
}
