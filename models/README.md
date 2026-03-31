# Python Pipeline

## Setup

```bash
cd python
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
