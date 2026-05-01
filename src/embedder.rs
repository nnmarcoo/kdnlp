use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::session::Session;
use ort::value::Tensor;

use crate::typing::Session as TypingSession;

const EMBED_DIM: usize = 128;

struct Model {
    session: Mutex<Session>,
    means: [f32; 3],
    stds: [f32; 3],
}

static MODEL: OnceLock<Option<Model>> = OnceLock::new();

/// Load the ONNX model and norm stats from disk. Call once at startup.
/// `model_dir` should contain `embedder.onnx` and `norm_stats.json`.
pub fn load(model_dir: &Path) -> bool {
    MODEL
        .get_or_init(|| {
            eprintln!("embedder: trying {}", model_dir.display());

            let stats_path = model_dir.join("norm_stats.json");
            let stats_text = match std::fs::read_to_string(&stats_path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("embedder: can't read {:?}: {e}", stats_path);
                    return None;
                }
            };
            let stats: serde_json::Value = serde_json::from_str(&stats_text).ok()?;
            let means = parse_f32_3(&stats["means"])?;
            let stds = parse_f32_3(&stats["stds"])?;

            // Explicitly load the ORT library so ort doesn't hang searching
            // system paths. Try next to the exe first, then the model dir.
            let lib_name = if cfg!(target_os = "windows") {
                "onnxruntime.dll"
            } else if cfg!(target_os = "macos") {
                "libonnxruntime.dylib"
            } else {
                "libonnxruntime.so"
            };
            let exe_lib = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join(lib_name)));
            let lib_path = exe_lib
                .filter(|p| p.exists())
                .unwrap_or_else(|| model_dir.join(lib_name));
            eprintln!("embedder: loading ORT lib from {:?}", lib_path);
            if let Err(e) = ort::init_from(&lib_path) {
                eprintln!("embedder: ORT lib load failed: {e}");
                return None;
            }

            let onnx_path = model_dir.join("embedder.onnx");
            eprintln!("embedder: loading ONNX from {:?}", onnx_path);
            let mut builder = match Session::builder() {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("embedder: session builder failed: {e}");
                    return None;
                }
            };
            let session = match builder.commit_from_file(&onnx_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("embedder: ONNX load failed: {e}");
                    return None;
                }
            };

            eprintln!("embedder: loaded OK");
            Some(Model {
                session: Mutex::new(session),
                means,
                stds,
            })
        })
        .is_some()
}

/// Embed a typing session into a 128-dim L2-normalized vector.
/// Returns None if the model isn't loaded or the session has no bigrams.
pub fn embed(typing: &TypingSession) -> Option<[f32; EMBED_DIM]> {
    let model = MODEL.get()?.as_ref()?;

    let seq = &typing.sequence;
    if seq.is_empty() {
        return None;
    }

    let length = seq.len();

    // Build (1, length, 3) float32 tensor, z-normalized — no length cap
    let mut data = vec![0f32; length * 3];
    for (i, &(_, iki, dwell, flight)) in seq.iter().enumerate() {
        let raw = [iki as f32, dwell as f32, flight as f32];
        for (j, (v, (&mean, &std))) in raw
            .iter()
            .zip(model.means.iter().zip(model.stds.iter()))
            .enumerate()
        {
            data[i * 3 + j] = (v - mean) / std;
        }
    }

    let x_tensor = Tensor::<f32>::from_array(([1usize, length, 3], data)).ok()?;
    let len_tensor = Tensor::<i64>::from_array(([1usize], vec![length as i64])).ok()?;

    let mut session = model.session.lock().ok()?;
    let outputs = session
        .run(ort::inputs![
            "keystrokes" => x_tensor,
            "lengths"    => len_tensor,
        ])
        .ok()?;

    let (_, flat) = outputs["embedding"].try_extract_tensor::<f32>().ok()?;

    let mut out = [0f32; EMBED_DIM];
    out.copy_from_slice(&flat[..EMBED_DIM]);
    Some(out)
}

/// Cosine similarity between two L2-normalized embeddings.
pub fn cosine_sim(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn parse_f32_3(v: &serde_json::Value) -> Option<[f32; 3]> {
    let arr = v.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    Some([
        arr[0].as_f64()? as f32,
        arr[1].as_f64()? as f32,
        arr[2].as_f64()? as f32,
    ])
}
