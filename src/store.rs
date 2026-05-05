use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::typing::Profile;

#[derive(Serialize, Deserialize)]
struct StoredKeyEvent {
    key: char,
    keycode: u32,
    press_ms: u64,
    release_ms: Option<u64>,
}

#[derive(Serialize, Deserialize)]
struct StoredProfile {
    name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    events: Vec<StoredKeyEvent>,
    bigrams: Vec<(char, char, f64)>,
    #[serde(default)]
    bigram_counts: Vec<(char, char, usize)>,
    #[serde(default)]
    char_count: usize,
    #[serde(default)]
    interval_count: usize,
    #[serde(default)]
    wpm: f64,
    #[serde(default)]
    avg_dwell_ms: f64,
    #[serde(default)]
    dwell_count: usize,
    #[serde(default = "default_session_count")]
    session_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
}

fn default_session_count() -> usize { 1 }

const DEMO_JSON: &str = include_str!("demo_profiles.json");

pub fn load_demo(n: usize) -> Vec<Profile> {
    match serde_json::from_str::<Vec<StoredProfile>>(DEMO_JSON) {
        Ok(stored) => stored.into_iter().take(n).map(stored_to_profile).collect(),
        Err(e) => {
            eprintln!("kdnlp: failed to parse demo profiles: {e}");
            Vec::new()
        }
    }
}

fn profiles_path() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("kdnlp").join("profiles.json"))
}

pub fn save(profiles: &[Profile]) {
    let Some(path) = profiles_path() else {
        eprintln!("kdnlp: could not determine data directory");
        return;
    };
    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        eprintln!("kdnlp: could not create data dir: {e}");
        return;
    }
    let stored: Vec<StoredProfile> = profiles.iter().map(profile_to_stored).collect();
    match serde_json::to_string_pretty(&stored) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                eprintln!("kdnlp: failed to write profiles: {e}");
            }
        }
        Err(e) => eprintln!("kdnlp: failed to serialize profiles: {e}"),
    }
}

pub fn load() -> Vec<Profile> {
    let Some(path) = profiles_path() else {
        return Vec::new();
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    match serde_json::from_str::<Vec<StoredProfile>>(&text) {
        Ok(stored) => stored.into_iter().map(stored_to_profile).collect(),
        Err(e) => {
            eprintln!("kdnlp: failed to parse profiles: {e}");
            Vec::new()
        }
    }
}

fn profile_to_stored(p: &Profile) -> StoredProfile {
    StoredProfile {
        name: p.name.clone(),
        events: Vec::new(),
        bigrams: p.bigrams.iter().map(|(&(a, b), &ms)| (a, b, ms)).collect(),
        bigram_counts: p
            .bigram_counts
            .iter()
            .map(|(&(a, b), &n)| (a, b, n))
            .collect(),
        char_count: p.char_count,
        interval_count: p.interval_count,
        wpm: p.wpm,
        avg_dwell_ms: p.avg_dwell_ms,
        dwell_count: p.dwell_count,
        session_count: p.session_count,
        embedding: p.embedding.as_ref().map(|e| e.to_vec()),
    }
}

fn stored_to_profile(s: StoredProfile) -> Profile {
    let bigrams: HashMap<(char, char), f64> = s
        .bigrams
        .into_iter()
        .map(|(a, b, ms)| ((a, b), ms))
        .collect();
    let bigram_counts: HashMap<(char, char), usize> = s
        .bigram_counts
        .into_iter()
        .map(|(a, b, n)| ((a, b), n))
        .collect();
    let char_count = if s.char_count > 0 {
        s.char_count
    } else {
        s.events.iter().filter(|e| e.key != '\x08').count()
    };
    let interval_count = if s.interval_count > 0 {
        s.interval_count
    } else {
        bigrams.len()
    };
    let (wpm, avg_dwell_ms, dwell_count) = if s.wpm > 0.0 {
        (s.wpm, s.avg_dwell_ms, s.dwell_count)
    } else {
        let wpm = if s.events.len() < 2 {
            0.0
        } else {
            let elapsed_ms = s
                .events
                .last()
                .unwrap()
                .press_ms
                .saturating_sub(s.events[0].press_ms) as f64;
            if elapsed_ms < 1.0 {
                0.0
            } else {
                (char_count as f64 / 5.0) / (elapsed_ms / 60_000.0)
            }
        };
        let dwells: Vec<f64> = s
            .events
            .iter()
            .filter_map(|e| e.release_ms.map(|r| (r - e.press_ms) as f64))
            .collect();
        let dwell_count = dwells.len();
        let avg_dwell_ms = if dwell_count > 0 {
            dwells.iter().sum::<f64>() / dwell_count as f64
        } else {
            0.0
        };
        (wpm, avg_dwell_ms, dwell_count)
    };
    let embedding = s.embedding.and_then(|v| {
        let arr: Box<[f32; 128]> = v.into_boxed_slice().try_into().ok()?;
        Some(arr)
    });
    Profile {
        name: s.name,
        bigrams,
        bigram_counts,
        char_count,
        interval_count,
        wpm,
        avg_dwell_ms,
        dwell_count,
        session_count: s.session_count,
        embedding,
    }
}
