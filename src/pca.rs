const DIM: usize = 128;

type ProjectResult = (Vec<(String, [f32; 2])>, Option<[f32; 2]>);

/// Project all profile embeddings + the current session embedding into 2D using
/// classical MDS, so 2D distances reflect actual cosine distances in embedding space.
pub fn project(profiles: &[(String, &[f32; DIM])], session: Option<&[f32; DIM]>) -> ProjectResult {
    let n_profiles = profiles.len();
    let n = n_profiles + session.is_some() as usize;

    if n < 2 {
        let labeled = profiles
            .iter()
            .map(|(name, _)| (name.clone(), [0.0f32; 2]))
            .collect();
        return (labeled, session.map(|_| [0.0f32; 2]));
    }

    // Collect all embeddings
    let mut embs: Vec<&[f32; DIM]> = profiles.iter().map(|(_, e)| *e).collect();
    if let Some(s) = session {
        embs.push(s);
    }

    // Build squared cosine distance matrix: d = 1 - cosine_sim, then squared
    let mut d2 = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let sim: f32 = embs[i].iter().zip(embs[j].iter()).map(|(a, b)| a * b).sum();
                let dist = (1.0 - sim.clamp(-1.0, 1.0)) as f64;
                d2[i * n + j] = dist * dist;
            }
        }
    }

    // Classical MDS: double-center the squared distance matrix
    // B = -0.5 * H * D2 * H  where H = I - (1/n) * 11^T
    let mut b = vec![0.0f64; n * n];
    let row_means: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| d2[i * n + j]).sum::<f64>() / n as f64)
        .collect();
    let total_mean: f64 = row_means.iter().sum::<f64>() / n as f64;
    for i in 0..n {
        for j in 0..n {
            let col_mean = (0..n).map(|k| d2[k * n + j]).sum::<f64>() / n as f64;
            b[i * n + j] = -0.5 * (d2[i * n + j] - row_means[i] - col_mean + total_mean);
        }
    }

    // Power iteration for top 2 eigenvectors of B
    let ev1 = top_eigenvector(&b, n, 100);
    let b_deflated = deflate(&b, n, &ev1);
    let ev2 = top_eigenvector(&b_deflated, n, 100);

    // Coordinates: scale by sqrt of eigenvalue
    let l1 = eigenvalue(&b, n, &ev1).max(0.0).sqrt();
    let l2 = eigenvalue(&b_deflated, n, &ev2).max(0.0).sqrt();

    let coords: Vec<[f32; 2]> = (0..n)
        .map(|i| [(ev1[i] * l1) as f32, (ev2[i] * l2) as f32])
        .collect();

    let labeled = profiles
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.clone(), coords[i]))
        .collect();

    let session_pt = session.map(|_| coords[n_profiles]);

    (labeled, session_pt)
}

fn mat_vec(b: &[f64], n: usize, v: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| b[i * n + j] * v[j]).sum())
        .collect()
}

fn top_eigenvector(b: &[f64], n: usize, iters: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    for _ in 0..iters {
        let w = mat_vec(b, n, &v);
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
        v = w.iter().map(|x| x / norm).collect();
    }
    v
}

fn eigenvalue(b: &[f64], n: usize, v: &[f64]) -> f64 {
    let bv = mat_vec(b, n, v);
    bv.iter().zip(v.iter()).map(|(x, y)| x * y).sum()
}

fn deflate(b: &[f64], n: usize, v: &[f64]) -> Vec<f64> {
    let lambda = eigenvalue(b, n, v);
    let mut out = b.to_vec();
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] -= lambda * v[i] * v[j];
        }
    }
    out
}
