<div align="center">
  <h1>kdnlp</h1>
  <p><em>keystroke dynamics identification using a neural network</em></p>

  ![License](https://img.shields.io/badge/license-MIT-0077aa?style=for-the-badge)
  ![This](https://img.shields.io/badge/this-is%20a%20demo-0077aa?style=for-the-badge)
</div>

---

A desktop app that identifies who is typing based on keystroke dynamics — the rhythm, timing, and cadence of how a person types. As you type a prompt, it captures inter-key intervals, dwell times, and flight times per bigram, embeds them using a pretrained BiLSTM, and ranks enrolled profiles by cosine similarity in real time.

Built with [Iced](https://iced.rs/) and [ONNX Runtime](https://onnxruntime.ai/).

## How it works

1. **Type** — start typing the prompt. Rankings and the embedding space update automatically after every keystroke (threshold: 5 bigrams).
2. **Enroll** — enter a name and press Enroll to save your typing profile. Profiles are persisted across sessions.
3. **Demo users** — load 5–500 pre-computed profiles from the Aalto dataset via the dropdown. These are held-out users never seen during training and are not persisted.

New users can be enrolled at any time without retraining.

## UI

- **Demo tab** — typing panel, real-time rankings card, and a 2D embedding space (MDS projection of cosine distances).
- **Profiles tab** — view, search, and delete enrolled profiles. Demo profiles are shown without a delete button.

## Model

Dual stacked bidirectional LSTM (TypeNet-inspired) trained with supervised contrastive loss on the [Aalto University Keystroke Dataset](https://userinterfaces.aalto.fi/136Mkeystrokes/) — 160,000 participants, 2,000 held-out for evaluation.

Features per bigram: `iki_ms`, `dwell_ms`, `flight_ms`. Output: 128-dim L2-normalized embedding.

**Evaluation on held-out users:**
- EER: ~4.1%
- AUC: 0.992
- EER plateaus at ~50 bigrams (~one sentence of typing)

See [`models/`](models/) for the full training and evaluation pipeline.

## Build

**Requirements**
- [Rust](https://www.rust-lang.org/tools/install)
- ONNX Runtime shared library (v1.24.x)

**Windows**

Place `onnxruntime.dll` in `lib/windows-x64/`. It will be copied next to the exe automatically.

```
cargo run --release
```

**Linux**

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz | tar xz
cp onnxruntime-linux-x64-1.24.4/lib/libonnxruntime.so lib/linux-x64/

cargo run --release
```

## Model files

`model/`:
- `embedder.onnx` — exported BiLSTM embedder
- `norm_stats.json` — per-feature z-normalization statistics

The 500 demo profiles are embedded in the binary at compile time (`src/demo_profiles.json`). To regenerate them after retraining, see [`models/`](models/).

---

*Accuracy improves with more enrolled sessions per user.*
