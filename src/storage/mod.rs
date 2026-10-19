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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_storage_lifecycle() -> Result<()> {
        let temp_dir = std::env::temp_dir().join(format!("test_event_storage_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
        let storage = LocalStorage::new(temp_dir.to_str().unwrap())?;

        let sample_key = "wedding_2026/photo_001.jpg";
        let sample_bytes = b"EXIF_JPEG_RAW_BYTES_TEST_12345";

        // 1. Write bytes
        let written_key = storage.write_bytes(sample_key, sample_bytes)?;
        assert_eq!(written_key, sample_key);

        // 2. Read bytes
        let read_back = storage.read_bytes(sample_key)?;
        assert_eq!(read_back, sample_bytes);

        // 3. Verify URL format
        let url = storage.get_url(sample_key);
        assert_eq!(url, "/static/photos/wedding_2026/photo_001.jpg");

        // 4. Nested directory auto-creation test
        let nested_key = "nested/subfolder/test.png";
        storage.write_bytes(nested_key, b"PNG_DATA")?;
        assert_eq!(storage.read_bytes(nested_key)?, b"PNG_DATA");

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
        Ok(())
    }

    #[test]
    fn test_s3_storage_url_formatting() -> Result<()> {
        let storage = S3Storage::new(
            "http://127.0.0.1:9000",
            "event-photos",
            Some("minioadmin".to_string()),
            Some("minioadmin".to_string()),
        )?;

        let key = "tech_summit/speaker.jpg";
        let url = storage.get_url(key);
        assert_eq!(url, "http://127.0.0.1:9000/event-photos/tech_summit/speaker.jpg");

        // Test with leading slash
        let url_leading = storage.get_url("/tech_summit/speaker.jpg");
        assert_eq!(url_leading, "http://127.0.0.1:9000/event-photos/tech_summit/speaker.jpg");

        Ok(())
    }
}

