use std::collections::{HashMap, HashSet};

use iced::Color;
use iced_plot::{PlotWidget, Series};

use crate::typing::{Profile, Session};

pub fn build_id_plot(session: &Session, profiles: &[Profile]) -> Option<PlotWidget> {
    if profiles.is_empty() {
        return None;
    }

    let session_avg = session.averaged();

    let all_bigrams: HashSet<(char, char)> = profiles
        .iter()
        .flat_map(|p| p.bigrams.keys().copied())
        .collect();

    if all_bigrams.len() < 2 {
        return None;
    }

    let mut variances: Vec<((char, char), f64)> = all_bigrams
        .iter()
        .map(|&bg| {
            let vals: Vec<f64> = profiles
                .iter()
                .filter_map(|p| p.bigrams.get(&bg).copied())
                .collect();
            let var = if vals.len() < 2 {
                0.0
            } else {
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64
            };
            (bg, var)
        })
        .collect();
    variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let ax = variances[0].0;
    let ay = variances[1].0;

    let global_mean: f64 = {
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

    let session_x = session_avg.get(&ax).copied().unwrap_or(global_mean);
    let session_y = session_avg.get(&ay).copied().unwrap_or(global_mean);

    let distances: Vec<f64> = profiles
        .iter()
        .map(|p| bigram_rms(&session_avg, &p.bigrams, global_mean))
        .collect();
    let max_dist = distances.iter().cloned().fold(0.0_f64, f64::max).max(1.0);

    let profile_positions: Vec<[f64; 2]> = profiles
        .iter()
        .map(|p| {
            [
                p.bigrams.get(&ax).copied().unwrap_or(global_mean),
                p.bigrams.get(&ay).copied().unwrap_or(global_mean),
            ]
        })
        .collect();

    let profile_colors: Vec<Color> = distances
        .iter()
        .map(|&d| {
            let t = (d / max_dist).clamp(0.0, 1.0) as f32;
            Color::from_rgb(0.38 + 0.54 * t, 0.82 - 0.47 * t, 0.48 - 0.13 * t)
        })
        .collect();

    let mut plot = PlotWidget::new();
    plot.set_x_axis_label(format!("{}{} ms", ax.0, ax.1));
    plot.set_y_axis_label(format!("{}{} ms", ay.0, ay.1));

    let _ = plot.add_series(
        Series::circles(profile_positions, 8.0)
            .with_label("profiles")
            .with_point_colors(profile_colors),
    );
    let _ = plot.add_series(
        Series::stars(vec![[session_x, session_y]], 14.0)
            .with_label("you")
            .with_color(Color::WHITE),
    );

    Some(plot)
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
