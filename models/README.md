# Python Pipeline

Training, evaluation, and export pipeline for the keystroke dynamics BiLSTM embedder.

## Setup

```bash
cd models
pip install -r requirements.txt
```

## Preprocessing

Parses the [Aalto University Keystroke Dataset](https://userinterfaces.aalto.fi/136Mkeystrokes/), extracts per-bigram timing features, writes train/test CSVs.

```bash
python preprocess.py --data_dir ../keystrokes/files --out_dir ./processed
```

Output: `processed/train.csv`, `processed/test.csv`. Each row: `participant_id`, `session_id`, `bigram`, `iki_ms`, `dwell_ms`, `flight_ms`. Takes ~10 minutes for all 168k files.

## Training

Dual stacked bidirectional LSTM with supervised contrastive loss (SupCon). L2-normalized embeddings allow open-enrollment without retraining.

```bash
python lstm.py --save_path D:/my_model --data_dir ./processed
```

Key options:
- `--n_train_users 160000` — users to train on (rest are held out)
- `--n_eval_users 2000` — held-out users for EER evaluation
- `--epochs 60`
- `--hidden_size 320`
- `--dropout 0.3` / `--recurrent_dropout 0.2`
- `--lr 0.002`
- `--n_augments 10` — random crops per session
- `--temperature 0.15` — SupCon loss temperature

Trains on GPU if available. ~1–2 hours on a modern GPU for the full dataset.

## Hyperparameter Tuning

```bash
python tune.py --stage 1 --n_trials 20   # broad search
python tune.py --stage 2 --n_trials 20   # focused search around best from stage 1
```

## Evaluation

Answers three research questions with charts saved to `--out_dir`:

- **Q1** — ROC curve, EER, AUC, and Rank-1 accuracy on 2,000 held-out users
- **Q2** — EER and Rank-1 vs. sample length (5–100 bigrams)
- **Q3** — Transformer encoder trained under identical conditions, compared against the LSTM baseline

```bash
# Q1 + Q2 only (fast):
python evaluate.py --data_dir ./processed --model_dir D:/my_model --out_dir ./eval_results --skip_transformer

# Q1 + Q2 + Q3 (trains Transformer, ~20–30 min on GPU):
python evaluate.py --data_dir ./processed --model_dir D:/my_model --out_dir ./eval_results --trans_epochs 30

# Q3 only (skip Q1+Q2):
python evaluate.py --data_dir ./processed --model_dir D:/my_model --out_dir ./eval_results --skip_q1_q2 --trans_epochs 30
```

## ONNX Export

```bash
python export_onnx.py --model_dir D:/my_model
```

Output: `D:/my_model/embedder.onnx`. The forward pass is rewritten to remove `pack_padded_sequence` (unsupported by the ONNX exporter), replaced with explicit index gathering on the last real timestep.

## Demo Profile Generation

Generates profiles from held-out Aalto eval users for embedding in the Rust binary. Uses the same `seed=42` split as training so no eval users overlap with training.

```bash
python generate_profiles.py \
  --data_dir ../keystrokes/files \
  --model_dir D:/my_model \
  --processed_dir ./processed \
  --n 500 \
  --out ../src/demo_profiles.json
```

After regenerating, rebuild the Rust app (`cargo build`) to embed the new profiles.

## Baseline

Weighted nearest-neighbor over z-normalized bigram timing profiles. No training required.

```bash
python baseline_nn.py --n_users 50 100 250 500
```
