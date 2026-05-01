import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import roc_curve

TIMING_COLS = ["iki_ms", "dwell_ms", "flight_ms"]


class KeystrokeDataset(Dataset):
    def __init__(self, df, pid_to_label, max_len=50, n_augments=1, noise_std=0.05):
        self.sequences = []
        self.lengths = []
        self.labels = []

        grouped = df.groupby(["participant_id", "session_id"])

        for (pid, sid), group in grouped:
            if pid not in pid_to_label:
                continue

            seq = group[TIMING_COLS].values.astype(np.float32)
            label = pid_to_label[pid]

            for _ in range(n_augments):
                crop, length = self._crop(seq, max_len)
                if n_augments > 1 and noise_std > 0:
                    crop = crop + np.random.normal(0, noise_std, crop.shape).astype(np.float32)
                padded = np.zeros((max_len, len(TIMING_COLS)), dtype=np.float32)
                padded[:length] = crop
                self.sequences.append(padded)
                self.lengths.append(length)
                self.labels.append(label)

        self.labels = np.array(self.labels)

    @staticmethod
    def _crop(seq, max_len):
        length = len(seq)
        if length <= max_len:
            return seq, length
        start = random.randint(0, length - max_len)
        return seq[start:start + max_len], max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx])
        length = torch.tensor(self.lengths[idx])
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, length, label


class PKSampler(Sampler):
    """
    For each batch: pick P users at random, then sample K sequences from
    each user, giving a batch of P*K samples. This guarantees every batch
    has multiple same-user pairs for hard negative mining to work on.
    """
    def __init__(self, labels, p, k):
        self.labels = np.array(labels)
        self.p = p
        self.k = k
        self.label_to_indices = {}
        for i, lbl in enumerate(self.labels):
            self.label_to_indices.setdefault(lbl, []).append(i)
        # Only keep users that have at least 2 samples (need a positive pair)
        self.valid_labels = [l for l, ids in self.label_to_indices.items() if len(ids) >= 2]

    def __len__(self):
        return (len(self.valid_labels) // self.p) * self.p * self.k

    def __iter__(self):
        labels = self.valid_labels.copy()
        random.shuffle(labels)
        for i in range(0, len(labels) - self.p + 1, self.p):
            batch_labels = labels[i:i + self.p]
            indices = []
            for lbl in batch_labels:
                pool = self.label_to_indices[lbl]
                chosen = random.choices(pool, k=self.k)
                indices.extend(chosen)
            yield indices


class KeystrokeEmbedder(nn.Module):
    """
    Dual stacked bidirectional LSTM that maps a variable-length keystroke
    sequence to a fixed-size L2-normalized embedding vector.

    Uses variational recurrent dropout between the two LSTM layers.
    No classification head — embeddings are compared directly via cosine
    similarity, so new users can be enrolled without retraining.
    """
    def __init__(self, input_size, hidden_size=128, embed_dim=128, dropout=0.5, recurrent_dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        self.fc = nn.Linear(hidden_size * 2, embed_dim)
        self.recurrent_dropout = recurrent_dropout

    def _variational_dropout(self, x):
        if not self.training or self.recurrent_dropout == 0:
            return x
        # Same mask across all timesteps so the dropout is consistent per sequence
        mask = torch.bernoulli(
            torch.ones(x.size(0), 1, x.size(2), device=x.device) * (1 - self.recurrent_dropout)
        ) / (1 - self.recurrent_dropout)
        return x * mask

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out1, _ = self.lstm1(packed)
        out1, _ = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        out1 = self._variational_dropout(out1)
        out1 = self.dropout(out1)

        actual_lengths = lengths.cpu().clamp(max=out1.size(1))
        packed2 = nn.utils.rnn.pack_padded_sequence(
            out1, actual_lengths, batch_first=True, enforce_sorted=False
        )
        _, (hn2, _) = self.lstm2(packed2)
        out = torch.cat((hn2[-2], hn2[-1]), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return F.normalize(out, p=2, dim=1)


def supcon_loss(embeddings, labels, temperature=0.07):
    """
    Supervised contrastive loss (Khosla et al. 2020).
    For each anchor, treat all same-label samples as positives and all
    different-label samples as negatives. The softmax over the full batch
    prevents embedding collapse because the model must spread embeddings
    apart to minimize the cross-entropy — there is no zero-gradient plateau.
    Temperature controls how tightly clusters form (lower = harder).
    """
    n = embeddings.size(0)
    sim = (embeddings @ embeddings.T) / temperature  # (N, N) cosine sims scaled

    # Mask out self-similarity on diagonal
    mask_self = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    same = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~mask_self

    # Need at least one positive per anchor
    has_pos = same.any(dim=1)
    if not has_pos.any():
        return embeddings.sum() * 0.0

    sim.masked_fill_(mask_self, float("-inf"))

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    # Sum log-prob only over positive positions (avoid -inf * 0 = NaN)
    pos_count = same.float().sum(dim=1).clamp(min=1)
    pos_log_prob = (log_prob.masked_fill(~same, 0.0)).sum(dim=1) / pos_count
    loss = -pos_log_prob[has_pos].mean()
    return loss


def compute_eer(model, enroll_loader, verify_loader, device):
    """
    Equal Error Rate: the threshold where false accept rate == false reject rate.
    Lower is better. Used instead of rank-1 accuracy because it measures
    authentication quality for open-set scenarios.
    """
    model.eval()
    enroll_embs, enroll_labels = [], []
    verify_embs, verify_labels = [], []

    with torch.no_grad():
        for x, lengths, labels in enroll_loader:
            enroll_embs.append(model(x.to(device), lengths).cpu())
            enroll_labels.append(labels)
        for x, lengths, labels in verify_loader:
            verify_embs.append(model(x.to(device), lengths).cpu())
            verify_labels.append(labels)

    enroll_embs = torch.cat(enroll_embs)
    enroll_labels = torch.cat(enroll_labels)
    verify_embs = torch.cat(verify_embs)
    verify_labels = torch.cat(verify_labels)

    # Build one profile per user
    unique_labels = enroll_labels.unique()
    profiles = torch.stack([
        enroll_embs[enroll_labels == lbl].mean(dim=0) for lbl in unique_labels
    ])
    profiles = F.normalize(profiles, p=2, dim=1)

    # Similarity scores between each verify sample and its genuine profile
    # and against all impostor profiles
    scores = []
    gt = []
    for emb, lbl in zip(verify_embs, verify_labels):
        emb = F.normalize(emb.unsqueeze(0), p=2, dim=1)
        sims = (emb @ profiles.T).squeeze(0)
        for j, profile_lbl in enumerate(unique_labels):
            scores.append(sims[j].item())
            gt.append(1 if profile_lbl == lbl else 0)

    scores = np.array(scores)
    gt = np.array(gt)
    fpr, tpr, _ = roc_curve(gt, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


def train_one_epoch(model, loader, optimizer, temperature, device):
    model.train()
    total_loss = 0.0
    total = 0

    for x, lengths, labels in loader:
        x, labels = x.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(x, lengths)
        loss = supcon_loss(embeddings, labels, temperature)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    return total_loss / total


def eval_loss(model, loader, temperature, device):
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for x, lengths, labels in loader:
            x, labels = x.to(device), labels.to(device)
            embeddings = model(x, lengths)
            loss = supcon_loss(embeddings, labels, temperature)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
    return total_loss / total if total > 0 else 0.0


def load_data(data_dir, n_train_users, n_eval_users, val_fraction=0.15, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print("Loading data...")
    train_df = pd.read_csv(f"{data_dir}/train.csv", encoding_errors="ignore")
    test_df = pd.read_csv(f"{data_dir}/test.csv", encoding_errors="ignore")

    all_users = train_df['participant_id'].unique().tolist()
    random.shuffle(all_users)

    total_needed = n_train_users + n_eval_users
    if total_needed > len(all_users):
        raise ValueError(
            f"Requested {n_train_users} train + {n_eval_users} eval = {total_needed} "
            f"but only {len(all_users)} available."
        )

    eval_users = set(all_users[:n_eval_users])
    train_users = all_users[n_eval_users:n_eval_users + n_train_users]
    print(f"Train users: {len(train_users)} | Held-out eval users: {len(eval_users)}")

    pid_to_label = {pid: i for i, pid in enumerate(train_users)}
    eval_pid_to_label = {pid: i for i, pid in enumerate(sorted(eval_users))}

    train_df_pool = train_df[train_df['participant_id'].isin(train_users)].copy()
    train_df_eval = train_df[train_df['participant_id'].isin(eval_users)].copy()
    test_df_eval  = test_df[test_df['participant_id'].isin(eval_users)].copy()

    print("Normalizing features...")
    means = train_df_pool[TIMING_COLS].mean()
    stds  = train_df_pool[TIMING_COLS].std()
    for df in [train_df_pool, train_df_eval, test_df_eval]:
        df[TIMING_COLS] = (df[TIMING_COLS] - means) / stds

    print("Splitting train/val sessions...")
    all_sessions = list(train_df_pool.groupby(["participant_id", "session_id"]).groups.keys())
    random.shuffle(all_sessions)
    n_val = max(1, int(len(all_sessions) * val_fraction))
    val_sessions = set(all_sessions[:n_val])

    val_mask = pd.Series(
        train_df_pool.set_index(["participant_id", "session_id"]).index.isin(val_sessions),
        index=train_df_pool.index
    )

    return {
        "fit_df":           train_df_pool[~val_mask],
        "val_df":           train_df_pool[val_mask],
        "train_df_eval":    train_df_eval,
        "test_df_eval":     test_df_eval,
        "pid_to_label":     pid_to_label,
        "eval_pid_to_label": eval_pid_to_label,
        "means":            means,
        "stds":             stds,
    }


def train(data, epochs=20, p_users=64, k_samples=8, hidden_size=320, lr=2e-3,
          dropout=0.3, recurrent_dropout=0.2, n_augments=10,
          embed_dim=128, temperature=0.15, max_len=50, seed=42,
          save_path=None, eer_every=5):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Building datasets...")
    train_dataset = KeystrokeDataset(data["fit_df"], data["pid_to_label"], max_len=max_len, n_augments=n_augments)
    val_dataset   = KeystrokeDataset(data["val_df"], data["pid_to_label"], max_len=max_len)
    eval_enroll   = KeystrokeDataset(data["train_df_eval"], data["eval_pid_to_label"], max_len=max_len)
    eval_verify   = KeystrokeDataset(data["test_df_eval"],  data["eval_pid_to_label"], max_len=max_len)

    # PK sampler: P users per batch, K samples each — guarantees same-user
    # pairs exist in every batch so hard negative mining has signal to work with
    batch_size = p_users * k_samples
    pk_sampler = PKSampler(train_dataset.labels, p=p_users, k=k_samples)
    train_loader       = DataLoader(train_dataset, batch_sampler=pk_sampler)
    val_loader         = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    eval_enroll_loader = DataLoader(eval_enroll,   batch_size=batch_size, shuffle=False)
    eval_verify_loader = DataLoader(eval_verify,   batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    model = KeystrokeEmbedder(
        input_size=len(TIMING_COLS),
        hidden_size=hidden_size,
        embed_dim=embed_dim,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, temperature, device)
        vl         = eval_loss(model, val_loader, temperature, device) if len(val_dataset) > 0 else 0.0
        scheduler.step(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        eer_str = ""
        if (epoch + 1) % eer_every == 0:
            eer = compute_eer(model, eval_enroll_loader, eval_verify_loader, device)
            eer_str = f" | EER: {eer*100:.2f}%"
            model.train()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {vl:.4f}{eer_str}")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    print("\nEvaluating on held-out users (open-enrollment simulation)...")
    eer = compute_eer(model, eval_enroll_loader, eval_verify_loader, device)
    print(f"Unseen users EER: {eer*100:.2f}%  (lower is better; random = 50%)")

    if save_path:
        from pathlib import Path
        Path(save_path).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/embedder.pt")
        with open(f"{save_path}/norm_stats.json", "w") as f:
            json.dump({"means": data["means"].tolist(), "stds": data["stds"].tolist()}, f)
        print(f"Saved model and norm stats to {save_path}/")

    return eer


def run(data_dir, n_train_users, n_eval_users, seed=42, epochs=20,
        p_users=64, k_samples=8, hidden_size=128, lr=1e-3, val_fraction=0.15,
        dropout=0.5, recurrent_dropout=0.2, n_augments=5,
        embed_dim=128, temperature=0.07, max_len=50, save_path=None, eer_every=5):
    data = load_data(data_dir, n_train_users, n_eval_users, val_fraction=val_fraction, seed=seed)
    return train(data, epochs=epochs, p_users=p_users, k_samples=k_samples,
                 hidden_size=hidden_size, lr=lr, dropout=dropout,
                 recurrent_dropout=recurrent_dropout, n_augments=n_augments,
                 embed_dim=embed_dim, temperature=temperature, max_len=max_len,
                 seed=seed, save_path=save_path, eer_every=eer_every)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",           default="./processed")
    parser.add_argument("--save_path",          default=None)
    parser.add_argument("--n_train_users",      type=int,   default=160000)
    parser.add_argument("--n_eval_users",       type=int,   default=2000)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--epochs",             type=int,   default=20)
    parser.add_argument("--p_users",            type=int,   default=64,  help="Users per batch")
    parser.add_argument("--eer_every",          type=int,   default=5,   help="Compute EER every N epochs")
    parser.add_argument("--k_samples",          type=int,   default=8,   help="Samples per user per batch")
    parser.add_argument("--hidden_size",        type=int,   default=320)
    parser.add_argument("--embed_dim",          type=int,   default=128)
    parser.add_argument("--lr",                 type=float, default=2e-3)
    parser.add_argument("--val_fraction",       type=float, default=0.15)
    parser.add_argument("--dropout",            type=float, default=0.3)
    parser.add_argument("--recurrent_dropout",  type=float, default=0.2)
    parser.add_argument("--n_augments",         type=int,   default=10)
    parser.add_argument("--temperature",        type=float, default=0.15)
    parser.add_argument("--max_len",            type=int,   default=50)
    args = parser.parse_args()
    run(args.data_dir, args.n_train_users, args.n_eval_users, args.seed,
        args.epochs, args.p_users, args.k_samples, args.hidden_size, args.lr,
        args.val_fraction, args.dropout, args.recurrent_dropout,
        args.n_augments, args.embed_dim, args.temperature, args.max_len,
        args.save_path, args.eer_every)
