use crate::embedder;
use crate::typing::{Profile, Session};

pub fn rank_profiles(session: &Session, profiles: &[Profile]) -> Vec<(String, f64)> {
    neural_rank(session, profiles)
}

fn neural_rank(session: &Session, profiles: &[Profile]) -> Vec<(String, f64)> {
    let Some(query_emb) = embedder::embed(session) else {
        return profiles.iter().map(|p| (p.name.clone(), 0.0)).collect();
    };

    let mut ranked: Vec<(String, f64)> = profiles
        .iter()
        .map(|p| {
            let score = p.embedding.as_ref()
                .map(|e| embedder::cosine_sim(&query_emb, e) as f64)
                .unwrap_or(0.0);
            (p.name.clone(), score)
        })
        .collect();
    // Higher cosine similarity = better match, sort descending
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

