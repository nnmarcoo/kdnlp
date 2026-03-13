<div align="center">
  <h1>kdnlp</h1>
  <p><em>keystroke dynamics profiling and identification</em></p>

  ![License](https://img.shields.io/badge/license-MIT-0077aa?style=for-the-badge)
  ![This](https://img.shields.io/badge/this-is%20a%20demo-0077aa?style=for-the-badge)
</div>

---

Records bigram flight times and dwell durations as you type prompts, builds per-user profiles, and plots session distance against stored profiles to identify who is typing. Built with [Iced](https://iced.rs/).

## Dataset

Training uses the [Aalto University Keystroke Dataset](https://userinterfaces.aalto.fi/136Mkeystrokes/), a large-scale collection of keystroke timings from over 168,000 participants. Each record contains per-key press and release timestamps, participant ID, and the typed sentence.

## Model

> TODO

## Build

**Requirements**

- [Rust](https://www.rust-lang.org/tools/install)

```
cargo run --release
```

---

*This is a demonstration. Accuracy improves with more enrolled profiles and longer typing samples.*
