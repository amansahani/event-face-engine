use anyhow::Result;
use std::fs;
use std::path::PathBuf;

pub trait StorageBackend: Send + Sync {
    fn read_bytes(&self, key: &str) -> Result<Vec<u8>>;
    fn write_bytes(&self, key: &str, data: &[u8]) -> Result<String>;
    fn get_url(&self, key: &str) -> String;
}

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
