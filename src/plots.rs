use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::typing::{Profile, Session};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentificationMethod {
    FlightTime,
}

impl fmt::Display for IdentificationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdentificationMethod::FlightTime => write!(f, "Flight Time"),
        }
    }
}

pub const METHODS: &[IdentificationMethod] = &[IdentificationMethod::FlightTime];

pub fn rank_profiles(
    method: IdentificationMethod,
    session: &Session,
    profiles: &[Profile],
) -> Vec<(String, f64)> {
    match method {
        IdentificationMethod::FlightTime => flight_time_rank(session, profiles),
    }
}

fn flight_time_rank(session: &Session, profiles: &[Profile]) -> Vec<(String, f64)> {
    if profiles.is_empty() {
        return Vec::new();
    }

    let session_avg = session.averaged();
    let global_mean = {
        let all: Vec<f64> = profiles
            .iter()
            .flat_map(|p| p.bigrams.values().copied())
            .collect();
        if all.is_empty() {
            200.0
        } else {
            all.iter().sum::<f64>() / all.len() as f64
        }
    };

    let mut ranked: Vec<(String, f64)> = profiles
        .iter()
        .map(|p| {
            (
                p.name.clone(),
                bigram_rms(&session_avg, &p.bigrams, global_mean),
            )
        })
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

fn bigram_rms(
    a: &HashMap<(char, char), f64>,
    b: &HashMap<(char, char), f64>,
    fallback: f64,
) -> f64 {
    let keys: HashSet<_> = a.keys().chain(b.keys()).copied().collect();
    if keys.is_empty() {
        return 0.0;
    }
    let sq: f64 = keys
        .iter()
        .map(|k| {
            let av = a.get(k).copied().unwrap_or(fallback);
            let bv = b.get(k).copied().unwrap_or(fallback);
            (av - bv).powi(2)
        })
        .sum();
    (sq / keys.len() as f64).sqrt()
}
