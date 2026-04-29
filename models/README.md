# Python Pipeline

## Setup

```bash
cd models
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
```

## Preprocessing

Parses the Aalto keystroke dataset, extracts bigram timing features, and writes train/test CSVs.

```bash
python preprocess.py
```

Output: `processed/train.csv` and `processed/test.csv`

Takes ~10 minutes for 168K participant files.

## Baseline Evaluation

Runs z-normalized weighted Euclidean nearest-neighbor identification at multiple user pool sizes.

```bash
python baseline_svm.py
```

Options:
- `--n_users 50 100 250 500` — pool sizes to evaluate
- `--seed 42` — random seed for reproducibility
- `--data_dir ./processed` — path to preprocessed CSVs

## LSTM

```bash
python lstm.py
```
Options (defaults):
- `--n_users 500` — pool sizes to evaluate
- `--seed 42` — random seed for reproducibility
- `--data_dir ./processed` — path to preprocessed CSVs
- `--epochs 20` - number of training loops
- `--batch_size 64` - number of sequences processed at once
- `--lr 1e-3` - learning rate for the optimizer
