# Python Pipeline

Training, evaluation, and export pipeline for the keystroke dynamics BiLSTM embedder.

## Setup

```bash
cd models
pip install -r requirements.txt
```

## Preprocessing

Parses the [Aalto University Keystroke Dataset](https://userinterfaces.aalto.fi/136Mkeystrokes/), extracts per-bigram timing features, and writes train/test CSVs.

```bash
python preprocess.py --data_dir ../keystrokes/files --out_dir ./processed
```

Output: `processed/train.csv` and `processed/test.csv`

Each row contains `participant_id`, `session_id`, `bigram`, `iki_ms`, `dwell_ms`, `flight_ms`.

Takes ~10 minutes for all 168k participant files.

## Training

Trains a dual stacked bidirectional LSTM with supervised contrastive loss. Embeddings are L2-normalized so new users can be enrolled at inference time without retraining.

```bash
python lstm.py --save_path ../model
```

Key options (defaults are tuned from hyperparameter search):
- `--n_train_users 160000` — users to train on
- `--n_eval_users 2000` — held-out users for open-enrollment EER evaluation
- `--epochs 20`
- `--hidden_size 320`
- `--dropout 0.3`
- `--recurrent_dropout 0.2`
- `--lr 0.002`
- `--n_augments 10` — random crops per session for data augmentation
- `--temperature 0.15` — supervised contrastive loss temperature
- `--save_path` — directory to save `embedder.pt` and `norm_stats.json`

Trains on GPU if available. Expects ~1-2 hours on a modern GPU for the full dataset.

## Hyperparameter Tuning

Two-stage random search over the training hyperparameters.

```bash
# Stage 1: broad search
python tune.py --stage 1 --n_trials 20

# Stage 2: focused search around best config from stage 1
python tune.py --stage 2 --n_trials 20
```

## Baseline

Weighted nearest-neighbor over z-normalized bigram timing profiles. No training required — useful as a reference point.

```bash
python baseline_nn.py --n_users 50 100 250 500
```

## ONNX Export

Converts the trained PyTorch model to ONNX for use in the Rust app.

```bash
python export_onnx.py --model_dir ../model
```

Output: `../model/embedder.onnx`

The export rewrites the forward pass to remove `pack_padded_sequence` (not supported by the ONNX exporter) and replaces it with explicit index gathering on the last real timestep.
