use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::typing::{KeyEvent, Profile};

// Profiles are stored as JSON at the platform data directory.
// Profile.bigrams uses tuple keys which are not valid JSON object keys,
// so bigrams are serialized as an array of (first, second, ms) triples.

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
    events: Vec<StoredKeyEvent>,
    bigrams: Vec<(char, char, f64)>,
}

fn profiles_path() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("kdnlp").join("profiles.json"))
}

pub fn save(profiles: &[Profile]) {
    let Some(path) = profiles_path() else {
        eprintln!("kdnlp: could not determine data directory");
        return;
    };
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("kdnlp: could not create data dir: {e}");
            return;
        }
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
        events: p
            .events
            .iter()
            .map(|e| StoredKeyEvent {
                key: e.key,
                keycode: e.keycode,
                press_ms: e.press_ms,
                release_ms: e.release_ms,
            })
            .collect(),
        bigrams: p.bigrams.iter().map(|(&(a, b), &ms)| (a, b, ms)).collect(),
    }
}

fn stored_to_profile(s: StoredProfile) -> Profile {
    Profile {
        name: s.name,
        events: s
            .events
            .into_iter()
            .map(|e| KeyEvent {
                key: e.key,
                keycode: e.keycode,
                press_ms: e.press_ms,
                release_ms: e.release_ms,
            })
            .collect(),
        bigrams: s
            .bigrams
            .into_iter()
            .map(|(a, b, ms)| ((a, b), ms))
            .collect::<HashMap<_, _>>(),
    }
}
