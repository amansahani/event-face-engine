pub mod config;
pub mod db;
pub mod engine;
pub mod storage;

use anyhow::Result;
use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use config::AppConfig;
use db::sqlite_store::SqliteVectorStore;
use engine::FaceEngine;
use serde::Serialize;
use std::sync::Arc;
use storage::{LocalStorage, StorageBackend};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

#[derive(Clone)]
struct AppState {
    db: SqliteVectorStore,
    engine: Arc<FaceEngine>,
    storage: Arc<dyn StorageBackend>,
    config: AppConfig,
}

#[derive(Serialize)]
struct FaceDetail {
    bbox: [i32; 4],
    score: f32,
    model_used: String,
}

#[derive(Serialize)]
struct UploadResponse {
    status: String,
    photo_key: String,
    url: String,
    faces_detected: usize,
    faces: Vec<FaceDetail>,
}

#[derive(Serialize)]
struct SearchResultItem {
    photo_key: String,
    url: String,
    similarity: f32,
    bbox: Vec<i32>,
}

#[derive(Serialize)]
struct SearchResponse {
    event_id: String,
    matched_count: usize,
    results: Vec<SearchResultItem>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // 1. Load Configuration
    let config = AppConfig::default();
    tracing::info!("Starting Event Face Engine with config: {:?}", config);

    // 2. Initialize Hardware-Agnostic Multi-Model Face Engine
    let engine = Arc::new(FaceEngine::new(
        &config.detector_model_path,
        &config.primary_model_path,
        &config.fast_model_path,
        config.prefer_device,
        config.intra_threads,
    )?);

    // 3. Initialize SQLite Vector Store
    let db = SqliteVectorStore::new(&config.db_path)?;

    // 4. Initialize Storage Backend
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(&config.photos_dir)?);

    let state = AppState {
        db,
        engine,
        storage,
        config: config.clone(),
    };

    // 5. Build Axum Router
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/api/events/:event_id/upload", post(upload_handler))
        .route("/api/events/:event_id/selfie_search", post(search_handler))
        .route("/", get(portal_handler))
        .nest_service("/static/photos", ServeDir::new(&config.photos_dir))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", config.server_host, config.server_port);
    tracing::info!("🚀 Event Face Engine (Multi-Model High Accuracy) listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "event_face_engine_rust",
        "pipeline": "SCRFD Multi-Face Detection + ArcFace/VGGFace2 Dynamic Routing",
        "runtime": state.engine.execution_target,
        "threads": state.config.intra_threads,
        "low_light_threshold": state.config.low_light_threshold,
        "similarity_threshold": state.config.similarity_threshold,
        "storage_dir": state.config.photos_dir,
    }))
}

async fn upload_handler(
    Path(event_id): Path<String>,
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    while let Some(field) = multipart.next_field().await.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))? {
        let file_name = field
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unnamed.jpg".to_string());

        let data = field.bytes().await.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        let photo_key = format!("{}/{}", event_id, file_name);

        // 1. Write raw photo to storage
        state.storage.write_bytes(&photo_key, &data).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        // 2. Decode image
        let img = FaceEngine::decode_image(&data)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Image decode error: {}", e)))?;
        let (orig_w, orig_h) = (img.width(), img.height());

        // 3. Upstream Multi-Face Detection via SCRFD
        let detected = state.engine.detect_faces(&img)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Detection error: {}", e)))?;

        // If no faces detected by SCRFD (e.g., extreme portrait close-up or fallback), treat full image as candidate
        let face_boxes = if detected.is_empty() {
            vec![[0i32, 0i32, orig_w as i32, orig_h as i32]]
        } else {
            detected.iter().map(|d| d.bbox).collect::<Vec<_>>()
        };

        let mut ingested_faces = Vec::new();

        // 4. Process each face individually with dynamic low-light & ArcFace recognition
        for bbox in face_boxes {
            let (embedding, model_type) = state.engine.extract_embedding(&img, bbox, state.config.low_light_threshold)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Embedding extraction error: {}", e)))?;

            state.db.insert_face(&event_id, &photo_key, &bbox, &embedding, None)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB insert error: {}", e)))?;

            ingested_faces.push(FaceDetail {
                bbox,
                score: 1.0,
                model_used: format!("{:?}", model_type),
            });
        }

        return Ok(Json(UploadResponse {
            status: "ingested".to_string(),
            photo_key: photo_key.clone(),
            url: state.storage.get_url(&photo_key),
            faces_detected: ingested_faces.len(),
            faces: ingested_faces,
        }));
    }

    Err((StatusCode::BAD_REQUEST, "No file provided in payload".to_string()))
}

async fn search_handler(
    Path(event_id): Path<String>,
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    while let Some(field) = multipart.next_field().await.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))? {
        let data = field.bytes().await.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

        // 1. Decode selfie
        let img = FaceEngine::decode_image(&data)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Selfie decode error: {}", e)))?;
        let (orig_w, orig_h) = (img.width(), img.height());

        // 2. Detect face in selfie
        let detected = state.engine.detect_faces(&img)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Detection error: {}", e)))?;

        let search_box = if let Some(first) = detected.first() {
            first.bbox
        } else {
            [0i32, 0i32, orig_w as i32, orig_h as i32]
        };

        // 3. Extract high-accuracy probe embedding (with low-light adaptive curve)
        let (query_emb, _) = state.engine.extract_embedding(&img, search_box, state.config.low_light_threshold)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Inference error: {}", e)))?;

        // 4. Sub-millisecond Dot-Product Search in SQLite with hardware SIMD
        let matches = state.db.search_similar(&event_id, &query_emb, state.config.similarity_threshold, 100)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Search error: {}", e)))?;

        let results = matches
            .into_iter()
            .map(|m| SearchResultItem {
                url: state.storage.get_url(&m.photo_key),
                photo_key: m.photo_key,
                similarity: (m.similarity * 10000.0).round() / 10000.0,
                bbox: m.bbox,
            })
            .collect::<Vec<_>>();

        return Ok(Json(SearchResponse {
            event_id,
            matched_count: results.len(),
            results,
        }));
    }

    Err((StatusCode::BAD_REQUEST, "No selfie provided in payload".to_string()))
}

async fn portal_handler() -> Html<&'static str> {
    Html(r#"
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>High-Accuracy Rust Event Face Portal</title>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-zinc-950 text-white min-h-screen flex flex-col items-center justify-center p-6">
      <div class="max-w-md w-full bg-zinc-900 rounded-3xl p-8 shadow-2xl border border-zinc-800 text-center">
        <div class="inline-block px-3 py-1 bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-mono rounded-full mb-3">🦀 SCRFD Multi-Face + ArcFace W600K</div>
        <h1 class="text-3xl font-extrabold mb-2 text-white">Find My Event Photos</h1>
        <p class="text-zinc-400 text-sm mb-6">High-accuracy attendee face matching with night & crowd support.</p>
        
        <input type="text" id="event_id" value="wedding_2026" class="w-full bg-zinc-950 border border-zinc-800 rounded-2xl px-4 py-3 mb-4 text-center font-mono text-sm focus:outline-none focus:border-amber-500">
        
        <label class="cursor-pointer block bg-amber-500 hover:bg-amber-400 text-zinc-950 font-bold py-3.5 px-6 rounded-2xl transition shadow-lg shadow-amber-500/20 mb-4">
          <span>📸 Take or Upload Selfie</span>
          <input type="file" id="selfie_input" accept="image/*" capture="user" class="hidden">
        </label>

        <div id="loading" class="hidden text-amber-400 font-semibold my-4 animate-pulse">Running sub-millisecond Rust matching...</div>
        <div id="results" class="grid grid-cols-2 gap-3 mt-6"></div>
      </div>

      <script>
        const input = document.getElementById('selfie_input');
        const eventIdInput = document.getElementById('event_id');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        input.addEventListener('change', async (e) => {
          if (!e.target.files.length) return;
          const file = e.target.files[0];
          const formData = new FormData();
          formData.append('selfie', file);

          loading.classList.remove('hidden');
          results.innerHTML = '';

          try {
            const res = await fetch(`/api/events/${eventIdInput.value}/selfie_search`, {
              method: 'POST',
              body: formData
            });
            const data = await res.json();
            loading.classList.add('hidden');

            if (data.results && data.results.length > 0) {
              data.results.forEach(item => {
                const img = document.createElement('img');
                img.src = item.url;
                img.className = 'w-full h-32 object-cover rounded-xl border border-zinc-800 shadow';
                results.appendChild(img);
              });
            } else {
              results.innerHTML = '<div class="col-span-2 text-zinc-500 text-sm">No photos found matching your selfie.</div>';
            }
          } catch (err) {
            loading.classList.add('hidden');
            alert('Search failed: ' + err.message);
          }
        });
      </script>
    </body>
    </html>
    "#)
}
