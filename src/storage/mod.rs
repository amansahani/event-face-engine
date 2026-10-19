use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;

pub trait StorageBackend: Send + Sync {
    fn read_bytes(&self, key: &str) -> Result<Vec<u8>>;
    fn write_bytes(&self, key: &str, data: &[u8]) -> Result<String>;
    fn get_url(&self, key: &str) -> String;
}

/// 1. Local Filesystem / Mounted NAS / HDD Storage Backend
pub struct LocalStorage {
    root_dir: PathBuf,
}

impl LocalStorage {
    pub fn new(root_dir: &str) -> Result<Self> {
        let path = PathBuf::from(root_dir);
        fs::create_dir_all(&path)?;
        Ok(Self { root_dir: path })
    }
}

impl StorageBackend for LocalStorage {
    fn read_bytes(&self, key: &str) -> Result<Vec<u8>> {
        let file_path = self.root_dir.join(key.trim_start_matches('/'));
        Ok(fs::read(file_path)?)
    }

    fn write_bytes(&self, key: &str, data: &[u8]) -> Result<String> {
        let file_path = self.root_dir.join(key.trim_start_matches('/'));
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&file_path, data)?;
        Ok(key.to_string())
    }

    fn get_url(&self, key: &str) -> String {
        format!("/static/photos/{}", key.trim_start_matches('/'))
    }
}

/// 2. Amazon S3 / MinIO / Cloudflare R2 S3-Compatible Storage Backend
pub struct S3Storage {
    client: reqwest::blocking::Client,
    endpoint: String,
    bucket: String,
    _access_key: Option<String>,
    _secret_key: Option<String>,
}

impl S3Storage {
    pub fn new(
        endpoint: &str,
        bucket: &str,
        access_key: Option<String>,
        secret_key: Option<String>,
    ) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let clean_endpoint = endpoint.trim_end_matches('/').to_string();

        Ok(Self {
            client,
            endpoint: clean_endpoint,
            bucket: bucket.to_string(),
            _access_key: access_key,
            _secret_key: secret_key,
        })
    }
}

impl StorageBackend for S3Storage {
    fn read_bytes(&self, key: &str) -> Result<Vec<u8>> {
        let clean_key = key.trim_start_matches('/');
        let url = format!("{}/{}/{}", self.endpoint, self.bucket, clean_key);
        let resp = self.client.get(&url).send().context("S3 GET request failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("S3 read failed with HTTP status: {}", resp.status());
        }
        Ok(resp.bytes()?.to_vec())
    }

    fn write_bytes(&self, key: &str, data: &[u8]) -> Result<String> {
        let clean_key = key.trim_start_matches('/');
        let url = format!("{}/{}/{}", self.endpoint, self.bucket, clean_key);
        let resp = self
            .client
            .put(&url)
            .header("Content-Type", "image/jpeg")
            .body(data.to_vec())
            .send()
            .context("S3 PUT request failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("S3 write failed with HTTP status: {}", resp.status());
        }
        Ok(key.to_string())
    }

    fn get_url(&self, key: &str) -> String {
        let clean_key = key.trim_start_matches('/');
        format!("{}/{}/{}", self.endpoint, self.bucket, clean_key)
    }
}
