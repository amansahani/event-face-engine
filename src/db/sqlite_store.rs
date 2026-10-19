use anyhow::Result;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorMatch {
    pub photo_key: String,
    pub similarity: f32,
    pub bbox: Vec<i32>,
    pub cluster_id: Option<i64>,
}

#[derive(Clone)]
pub struct SqliteVectorStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteVectorStore {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS event_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                photo_key TEXT NOT NULL,
                bbox TEXT NOT NULL,
                embedding BLOB NOT NULL,
                cluster_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_id ON event_faces(event_id);",
            [],
        )?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn insert_face(
        &self,
        event_id: &str,
        photo_key: &str,
        bbox: &[i32],
        embedding: &[f32],
        cluster_id: Option<i64>,
    ) -> Result<()> {
        let bbox_json = serde_json::to_string(bbox)?;
        // Convert &[f32] to byte slice
        let emb_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                embedding.as_ptr() as *const u8,
                embedding.len() * std::mem::size_of::<f32>(),
            )
        };

        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO event_faces (event_id, photo_key, bbox, embedding, cluster_id) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![event_id, photo_key, bbox_json, emb_bytes, cluster_id],
        )?;

        Ok(())
    }

    pub fn search_similar(
        &self,
        event_id: &str,
        query_embedding: &[f32],
        threshold: f32,
        top_k: usize,
    ) -> Result<Vec<VectorMatch>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT photo_key, bbox, embedding, cluster_id FROM event_faces WHERE event_id = ?1",
        )?;

        let mut matches: Vec<VectorMatch> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let rows = stmt.query_map(params![event_id], |row| {
            let photo_key: String = row.get(0)?;
            let bbox_str: String = row.get(1)?;
            let emb_bytes: Vec<u8> = row.get(2)?;
            let cluster_id: Option<i64> = row.get(3)?;
            Ok((photo_key, bbox_str, emb_bytes, cluster_id))
        })?;

        for r in rows {
            let (photo_key, bbox_str, emb_bytes, cluster_id) = r?;
            if seen.contains(&photo_key) {
                continue;
            }

            // Convert bytes back to &[f32]
            let emb_slice: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    emb_bytes.as_ptr() as *const f32,
                    emb_bytes.len() / std::mem::size_of::<f32>(),
                )
            };

            // Hardware-Accelerated Vector Dot-Product (AVX2/FMA, ARM NEON, or RISC-V)
            let similarity = crate::engine::simd::dot_product_simd(query_embedding, emb_slice);

            if similarity >= threshold {
                let bbox: Vec<i32> = serde_json::from_str(&bbox_str).unwrap_or_default();
                seen.insert(photo_key.clone());
                matches.push(VectorMatch {
                    photo_key,
                    similarity,
                    bbox,
                    cluster_id,
                });
            }
        }

        // Sort descending by similarity
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        matches.truncate(top_k);

        Ok(matches)
    }
}
