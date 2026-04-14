use std::collections::HashMap;

pub fn project_profiles(
    profiles: &[(String, HashMap<(char, char), f64>)],
    session: Option<&HashMap<(char, char), f64>>,
    global_mean: f64,
) -> (Vec<(String, [f32; 2])>, Option<[f32; 2]>) {
    let vocab: Vec<(char, char)> = {
        let mut keys: std::collections::HashSet<(char, char)> = profiles[0].1.keys().copied().collect();
        for (_, m) in profiles.iter().skip(1) {
            keys.retain(|k| m.contains_key(k));
        }
        if let Some(s) = session {
            keys.retain(|k| s.contains_key(k));
        }
        if keys.len() < 2 {
            // fall back to union if intersection is too sparse
            let mut union: std::collections::HashSet<(char, char)> = profiles
                .iter()
                .flat_map(|(_, m)| m.keys().copied())
                .collect();
            if let Some(s) = session {
                union.extend(s.keys().copied());
            }
            keys = union;
        }
        let mut v: Vec<_> = keys.into_iter().collect();
        v.sort_unstable();
        v
    };

    let d = vocab.len();
    if d < 2 {
        let labeled: Vec<_> = profiles
            .iter()
            .map(|(n, _)| (n.clone(), [0.0f32; 2]))
            .collect();
        return (labeled, session.map(|_| [0.0f32; 2]));
    }

    let to_vec = |map: &HashMap<(char, char), f64>| -> Vec<f64> {
        vocab
            .iter()
            .map(|k| map.get(k).copied().unwrap_or(global_mean))
            .collect()
    };

    let normalize = |mut v: Vec<f64>| -> Vec<f64> {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        v.iter_mut().for_each(|x| *x -= mean);
        v
    };

    let mut rows: Vec<Vec<f64>> = profiles.iter().map(|(_, m)| normalize(to_vec(m))).collect();
    let session_row = session.map(|s| normalize(to_vec(s)));
    if let Some(ref sr) = session_row {
        rows.push(sr.clone());
    }

    let n = rows.len();
    let means: Vec<f64> = (0..d)
        .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();

    for row in &mut rows {
        for (j, v) in row.iter_mut().enumerate() {
            *v -= means[j];
        }
    }

    let pc1 = top_eigenvector(&rows, d, 50);
    let deflated = deflate(&rows, &pc1);
    let pc2 = top_eigenvector(&deflated, d, 50);

    let project = |row: &Vec<f64>| -> [f32; 2] {
        let x = dot(row, &pc1) as f32;
        let y = dot(row, &pc2) as f32;
        [x, y]
    };

    let profile_count = profiles.len();
    let labeled: Vec<(String, [f32; 2])> = profiles
        .iter()
        .zip(rows.iter())
        .map(|((name, _), row)| (name.clone(), project(row)))
        .collect();

    let session_pt = session_row.as_ref().map(|_| project(&rows[profile_count]));

    (labeled, session_pt)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn top_eigenvector(rows: &[Vec<f64>], d: usize, iters: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..d).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    for _ in 0..iters {
        let mut w = vec![0.0f64; d];
        for row in rows {
            let s = dot(row, &v);
            for (j, wj) in w.iter_mut().enumerate() {
                *wj += s * row[j];
            }
        }
        let norm = dot(&w, &w).sqrt().max(1e-10);
        v = w.iter().map(|x| x / norm).collect();
    }
    v
}

fn deflate(rows: &[Vec<f64>], pc: &[f64]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            let s = dot(row, pc);
            row.iter().zip(pc.iter()).map(|(x, p)| x - s * p).collect()
        })
        .collect()
}
