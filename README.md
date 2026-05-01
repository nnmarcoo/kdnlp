<div align="center">
  <h1>kdnlp</h1>
  <p><em>keystroke dynamics identification using a neural network</em></p>

  ![License](https://img.shields.io/badge/license-MIT-0077aa?style=for-the-badge)
  ![This](https://img.shields.io/badge/this-is%20a%20demo-0077aa?style=for-the-badge)
</div>

---

A desktop app that identifies who is typing based on keystroke dynamics — the rhythm, timing, and cadence of how a person types. As you type a prompt, it captures inter-key intervals, dwell times, and flight times per bigram, embeds them using a pretrained BiLSTM, and compares the result against enrolled user profiles using cosine similarity.

Built with [Iced](https://iced.rs/) and [ONNX Runtime](https://onnxruntime.ai/).

## How it works

1. **Enroll** — type a prompt and enter your name. The app embeds your session into a 128-dimensional vector using the pretrained model and saves it as your profile.
2. **Identify** — type again and hit Identify. The app embeds the new session and ranks all enrolled profiles by cosine similarity. The closest match is shown as the predicted identity.

New users can be enrolled at any time without retraining the model.

## Model

A dual stacked bidirectional LSTM (TypeNet-inspired) trained with supervised contrastive loss on the [Aalto University Keystroke Dataset](https://userinterfaces.aalto.fi/136Mkeystrokes/) — 160,000 participants, open-enrollment evaluation.

Features per keystroke bigram: `iki_ms` (inter-key interval), `dwell_ms` (key hold time), `flight_ms` (release-to-press gap).

The trained model achieves ~5% Equal Error Rate on held-out users never seen during training.

See [`models/`](models/) for the full training pipeline.

## Build

**Requirements**

- [Rust](https://www.rust-lang.org/tools/install)
- ONNX Runtime shared library (see below)

**Windows**

Place `onnxruntime.dll` (v1.24.x) in `lib/windows-x64/`. It will be copied next to the exe automatically by the build script.

```
cargo run --release
```

**Linux**

```bash
# Download and place the ONNX Runtime library
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz | tar xz
cp onnxruntime-linux-x64-1.24.4/lib/libonnxruntime.so lib/linux-x64/

cargo run --release
```

The build script copies the library and model files next to the exe on every build. No environment variables required.

## Model files

The pretrained model lives in `model/`:
- `embedder.onnx` — the exported BiLSTM embedder
- `norm_stats.json` — per-feature z-normalization statistics from the training set

To retrain or export a new model, see [`models/`](models/).

---

*This is a demonstration. Accuracy improves with more enrolled sessions per user.*
