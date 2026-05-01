"""
evaluate.py — answers three research questions about keystroke dynamics identification.

Q1: Can the LSTM model accurately identify individuals? (Rank-1, EER, ROC curve)
Q2: How does accuracy decay as sample length decreases?
Q3: Does a Transformer encoder outperform the LSTM baseline?

Usage:
    python evaluate.py --data_dir ./processed --model_dir D:/kd_best --out_dir ./eval_results
    python evaluate.py --data_dir ./processed --model_dir D:/kd_best --out_dir ./eval_results --skip_transformer
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

from lstm import (
    TIMING_COLS, KeystrokeDataset, PKSampler,
    KeystrokeEmbedder, supcon_loss, compute_eer,
    train_one_epoch, eval_loss, load_data,
)

# ── colour palette ──────────────────────────────────────────────────────────
C_LSTM  = "#4C9BE8"
C_TRANS = "#E8824C"
C_GRID  = "#333333"

plt.rcParams.update({
    "figure.facecolor": "#1a1a1a",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#555555",
    "axes.labelcolor":  "#cccccc",
    "axes.titlecolor":  "#eeeeee",
    "xtick.color":      "#aaaaaa",
    "ytick.color":      "#aaaaaa",
    "text.color":       "#cccccc",
    "grid.color":       C_GRID,
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "legend.facecolor": "#2a2a2a",
    "legend.edgecolor": "#555555",
    "font.size":        11,
})


# ═══════════════════════════════════════════════════════════════════════════
# Transformer embedder
# ═══════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerEmbedder(nn.Module):
    """
    Transformer encoder that maps a variable-length keystroke sequence to a
    fixed-size L2-normalized embedding. Architecture mirrors the LSTM in
    parameter count for a fair comparison.
    """
    def __init__(self, input_size=3, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=512, embed_dim=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc      = nn.Linear(d_model, embed_dim)

    def forward(self, x, lengths):
        # Build key-padding mask: True where position is padding
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)  # (B, T)

        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=mask)

        # Mean-pool over real tokens
        lengths_f = lengths.float().clamp(min=1).unsqueeze(1)  # (B, 1)
        mask_f    = (~mask).float().unsqueeze(2)                # (B, T, 1)
        pooled    = (x * mask_f).sum(dim=1) / lengths_f        # (B, d_model)

        return F.normalize(self.fc(pooled), p=2, dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# Shared evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_embeddings(model, loader, device):
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for x, lengths, lbl in loader:
            embs.append(model(x.to(device), lengths.to(device)).cpu())
            labels.append(lbl)
    return torch.cat(embs), torch.cat(labels)


def rank1_accuracy(enroll_embs, enroll_labels, verify_embs, verify_labels):
    """Nearest-neighbour rank-1: does the closest enrolled profile match the query?"""
    unique = enroll_labels.unique()
    profiles = torch.stack([
        F.normalize(enroll_embs[enroll_labels == lbl].mean(0, keepdim=True), p=2, dim=1).squeeze(0)
        for lbl in unique
    ])
    sims    = verify_embs @ profiles.T          # (N_verify, N_users)
    preds   = unique[sims.argmax(dim=1)]
    correct = (preds == verify_labels).float().mean().item()
    return correct


def roc_data(enroll_embs, enroll_labels, verify_embs, verify_labels):
    unique = enroll_labels.unique()
    profiles = torch.stack([
        F.normalize(enroll_embs[enroll_labels == lbl].mean(0, keepdim=True), p=2, dim=1).squeeze(0)
        for lbl in unique
    ])
    scores, gt = [], []
    for emb, lbl in zip(verify_embs, verify_labels):
        sims = (emb.unsqueeze(0) @ profiles.T).squeeze(0)
        for j, pl in enumerate(unique):
            scores.append(sims[j].item())
            gt.append(1 if pl == lbl else 0)
    scores = np.array(scores)
    gt     = np.array(gt)
    fpr, tpr, _ = roc_curve(gt, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    return fpr, tpr, eer


# ═══════════════════════════════════════════════════════════════════════════
# Q1 — Identification accuracy
# ═══════════════════════════════════════════════════════════════════════════

def q1_accuracy(model, enroll_loader, verify_loader, device, out_dir, label="LSTM"):
    print(f"\n── Q1: Identification accuracy ({label}) ──")
    enroll_embs, enroll_labels = get_embeddings(model, enroll_loader, device)
    verify_embs,  verify_labels = get_embeddings(model, verify_loader,  device)

    r1  = rank1_accuracy(enroll_embs, enroll_labels, verify_embs, verify_labels)
    fpr, tpr, eer = roc_data(enroll_embs, enroll_labels, verify_embs, verify_labels)
    roc_auc = auc(fpr, tpr)

    print(f"  Rank-1 accuracy : {r1*100:.2f}%")
    print(f"  EER             : {eer*100:.2f}%")
    print(f"  AUC             : {roc_auc:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=C_LSTM, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#555555", lw=1, linestyle="--", label="Random")
    eer_x = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]
    ax.scatter([eer_x], [1 - eer_x], color="#ff6b6b", zorder=5, s=60,
               label=f"EER = {eer*100:.2f}%")
    ax.set_xlabel("False Accept Rate")
    ax.set_ylabel("True Accept Rate")
    ax.set_title(f"Q1 - ROC Curve\nRank-1: {r1*100:.1f}%  EER: {eer*100:.2f}%  AUC: {roc_auc:.3f}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    path = out_dir / f"q1_roc_{label.lower()}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")
    return r1, eer, roc_auc


# ═══════════════════════════════════════════════════════════════════════════
# Q2 — Accuracy vs sample length
# ═══════════════════════════════════════════════════════════════════════════

def q2_length_decay(model, data, device, out_dir,
                    lengths=(5, 10, 15, 20, 30, 50, 75, 100),
                    batch_size=512, label="LSTM"):
    print(f"\n── Q2: Accuracy vs sample length ({label}) ──")

    enroll_embs_full, enroll_labels = get_embeddings(
        model,
        DataLoader(
            KeystrokeDataset(data["train_df_eval"], data["eval_pid_to_label"], max_len=200),
            batch_size=batch_size, shuffle=False,
        ),
        device,
    )

    eers, r1s = [], []
    for max_len in lengths:
        ds = KeystrokeDataset(data["test_df_eval"], data["eval_pid_to_label"], max_len=max_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        verify_embs, verify_labels = get_embeddings(model, loader, device)

        _, _, eer = roc_data(enroll_embs_full, enroll_labels, verify_embs, verify_labels)
        r1 = rank1_accuracy(enroll_embs_full, enroll_labels, verify_embs, verify_labels)
        eers.append(eer * 100)
        r1s.append(r1 * 100)
        print(f"  max_len={max_len:4d}  EER={eer*100:.2f}%  Rank-1={r1*100:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(lengths, eers, color=C_LSTM if label == "LSTM" else C_TRANS,
             marker="o", lw=2, label=label)
    ax1.set_xlabel("Max bigrams used")
    ax1.set_ylabel("EER (%)")
    ax1.set_title("Q2 - EER vs Sample Length")
    ax1.invert_yaxis()
    ax1.grid(True)
    ax1.legend()

    ax2.plot(lengths, r1s, color=C_LSTM if label == "LSTM" else C_TRANS,
             marker="o", lw=2, label=label)
    ax2.set_xlabel("Max bigrams used")
    ax2.set_ylabel("Rank-1 Accuracy (%)")
    ax2.set_title("Q2 - Rank-1 Accuracy vs Sample Length")
    ax2.grid(True)
    ax2.legend()

    fig.suptitle(f"Accuracy decay as sample shortens — {label}", y=1.01)
    fig.tight_layout()
    path = out_dir / f"q2_length_decay_{label.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return list(lengths), eers, r1s


# ═══════════════════════════════════════════════════════════════════════════
# Q3 — Train Transformer and compare
# ═══════════════════════════════════════════════════════════════════════════

def q3_train_transformer(data, epochs, p_users, k_samples, device, out_dir,
                         lr=1e-3, dropout=0.1, embed_dim=128, temperature=0.07,
                         max_len=50, seed=42):
    print(f"\n── Q3: Training Transformer ({epochs} epochs) ──")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    batch_size = p_users * k_samples

    train_ds = KeystrokeDataset(data["fit_df"], data["pid_to_label"], max_len=max_len, n_augments=5)
    val_ds   = KeystrokeDataset(data["val_df"], data["pid_to_label"], max_len=max_len)

    pk = PKSampler(train_ds.labels, p=p_users, k=k_samples)
    train_loader = DataLoader(train_ds, batch_sampler=pk)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = TransformerEmbedder(
        input_size=len(TIMING_COLS), d_model=128, nhead=4,
        num_layers=3, dim_feedforward=512,
        embed_dim=embed_dim, dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_val, best_state = float("inf"), None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        tl = train_one_epoch(model, train_loader, optimizer, temperature, device)
        vl = eval_loss(model, val_loader, temperature, device)
        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch+1}/{epochs} | Train {tl:.4f} | Val {vl:.4f}")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, train_losses, val_losses


def q3_compare(lstm_model, trans_model, data, device, out_dir,
               lstm_train_losses, lstm_val_losses,
               trans_train_losses, trans_val_losses,
               max_len=50, batch_size=512):
    print("\n── Q3: Comparing LSTM vs Transformer ──")

    enroll_ds = KeystrokeDataset(data["train_df_eval"], data["eval_pid_to_label"], max_len=max_len)
    verify_ds  = KeystrokeDataset(data["test_df_eval"],  data["eval_pid_to_label"], max_len=max_len)
    enroll_loader = DataLoader(enroll_ds, batch_size=batch_size, shuffle=False)
    verify_loader  = DataLoader(verify_ds,  batch_size=batch_size, shuffle=False)

    results = {}
    for name, model, color in [("LSTM", lstm_model, C_LSTM), ("Transformer", trans_model, C_TRANS)]:
        ee, el = get_embeddings(model, enroll_loader, device)
        ve, vl = get_embeddings(model, verify_loader,  device)
        _, _, eer = roc_data(ee, el, ve, vl)
        r1 = rank1_accuracy(ee, el, ve, vl)
        results[name] = {"eer": eer * 100, "r1": r1 * 100, "color": color}
        print(f"  {name:12s}  EER={eer*100:.2f}%  Rank-1={r1*100:.1f}%")

    # ── Plot 1: training curves ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs_lstm  = range(1, len(lstm_train_losses) + 1)
    epochs_trans = range(1, len(trans_train_losses) + 1)

    ax1.plot(epochs_lstm,  lstm_train_losses,  color=C_LSTM,  lw=2, label="LSTM train")
    ax1.plot(epochs_lstm,  lstm_val_losses,    color=C_LSTM,  lw=2, linestyle="--", label="LSTM val")
    ax1.plot(epochs_trans, trans_train_losses, color=C_TRANS, lw=2, label="Transformer train")
    ax1.plot(epochs_trans, trans_val_losses,   color=C_TRANS, lw=2, linestyle="--", label="Transformer val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SupCon Loss")
    ax1.set_title("Q3 - Training Curves")
    ax1.legend()
    ax1.grid(True)

    # ── Plot 2: EER and Rank-1 bar chart ──
    names  = list(results.keys())
    eers   = [results[n]["eer"] for n in names]
    r1s    = [results[n]["r1"]  for n in names]
    colors = [results[n]["color"] for n in names]
    x = np.arange(len(names))
    w = 0.35

    ax2.bar(x - w/2, eers, w, color=colors, alpha=0.85, label="EER % (lower=better)")
    ax2.bar(x + w/2, r1s,  w, color=colors, alpha=0.45, label="Rank-1 % (higher=better)")
    for i, (e, r) in enumerate(zip(eers, r1s)):
        ax2.text(i - w/2, e + 0.3, f"{e:.1f}%", ha="center", va="bottom", fontsize=9)
        ax2.text(i + w/2, r + 0.3, f"{r:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("%")
    ax2.set_title("Q3 - LSTM vs Transformer")
    ax2.legend()
    ax2.grid(True, axis="y")

    fig.suptitle("Q3 · Does a Transformer outperform the LSTM?", y=1.01)
    fig.tight_layout()
    path = out_dir / "q3_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Combined Q2 overlay (both models on one chart)
# ═══════════════════════════════════════════════════════════════════════════

def plot_q2_overlay(lengths, lstm_eers, lstm_r1s, trans_eers, trans_r1s, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(lengths, lstm_eers,  color=C_LSTM,  marker="o", lw=2, label="LSTM")
    ax1.plot(lengths, trans_eers, color=C_TRANS, marker="s", lw=2, label="Transformer")
    ax1.set_xlabel("Max bigrams used")
    ax1.set_ylabel("EER (%)")
    ax1.set_title("Q2 - EER vs Sample Length")
    ax1.invert_yaxis()
    ax1.grid(True)
    ax1.legend()

    ax2.plot(lengths, lstm_r1s,  color=C_LSTM,  marker="o", lw=2, label="LSTM")
    ax2.plot(lengths, trans_r1s, color=C_TRANS, marker="s", lw=2, label="Transformer")
    ax2.set_xlabel("Max bigrams used")
    ax2.set_ylabel("Rank-1 Accuracy (%)")
    ax2.set_title("Q2 · Rank-1 vs Sample Length")
    ax2.grid(True)
    ax2.legend()

    fig.suptitle("Accuracy decay as sample shortens — LSTM vs Transformer", y=1.01)
    fig.tight_layout()
    path = out_dir / "q2_overlay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overlay → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def load_lstm(model_dir, device):
    model_dir = Path(model_dir)
    state = torch.load(model_dir / "embedder.pt", map_location="cpu", weights_only=True)
    hidden_size = state["lstm1.weight_ih_l0"].shape[0] // 4
    embed_dim   = state["fc.weight"].shape[0]
    model = KeystrokeEmbedder(
        input_size=len(TIMING_COLS),
        hidden_size=hidden_size,
        embed_dim=embed_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded LSTM  hidden={hidden_size}  embed={embed_dim}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",          default="./processed")
    parser.add_argument("--model_dir",         default="D:/kd_best")
    parser.add_argument("--out_dir",           default="./eval_results")
    parser.add_argument("--n_train_users",     type=int,   default=160000)
    parser.add_argument("--n_eval_users",      type=int,   default=2000)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--trans_epochs",      type=int,   default=30,
                        help="Epochs to train the Transformer for Q3")
    parser.add_argument("--p_users",           type=int,   default=32)
    parser.add_argument("--k_samples",         type=int,   default=8)
    parser.add_argument("--max_len",           type=int,   default=50)
    parser.add_argument("--batch_size",        type=int,   default=512)
    parser.add_argument("--skip_transformer",  action="store_true",
                        help="Skip Q3 Transformer training (Q1+Q2 only)")
    parser.add_argument("--skip_q1_q2",       action="store_true",
                        help="Skip Q1+Q2 and go straight to Q3 Transformer training")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data (same split as training — seed=42 guarantees same held-out users)
    data = load_data(args.data_dir, args.n_train_users, args.n_eval_users, seed=args.seed)

    enroll_ds = KeystrokeDataset(data["train_df_eval"], data["eval_pid_to_label"], max_len=args.max_len)
    verify_ds  = KeystrokeDataset(data["test_df_eval"],  data["eval_pid_to_label"], max_len=args.max_len)
    enroll_loader = DataLoader(enroll_ds, batch_size=args.batch_size, shuffle=False)
    verify_loader  = DataLoader(verify_ds,  batch_size=args.batch_size, shuffle=False)

    # ── Load LSTM ──
    lstm_model = load_lstm(args.model_dir, device)

    # Reconstruct LSTM training losses from a dummy (we don't have them saved)
    # Just use [] so the comparison chart skips the curve if not available
    lstm_train_losses, lstm_val_losses = [], []

    length_steps = (5, 10, 15, 20, 30, 50, 75, 100)

    if not args.skip_q1_q2:
        # ── Q1 ──
        q1_accuracy(lstm_model, enroll_loader, verify_loader, device, out_dir, label="LSTM")

        # ── Q2 ──
        lengths, lstm_eers, lstm_r1s = q2_length_decay(
            lstm_model, data, device, out_dir, lengths=length_steps,
            batch_size=args.batch_size, label="LSTM",
        )
    else:
        lengths, lstm_eers, lstm_r1s = list(length_steps), [], []

    if not args.skip_transformer:
        # ── Q3: train Transformer ──
        trans_model, trans_train_losses, trans_val_losses = q3_train_transformer(
            data, epochs=args.trans_epochs,
            p_users=args.p_users, k_samples=args.k_samples,
            device=device, out_dir=out_dir,
            max_len=args.max_len, seed=args.seed,
        )

        # Q1 for Transformer
        q1_accuracy(trans_model, enroll_loader, verify_loader, device, out_dir, label="Transformer")

        # Q2 for Transformer
        _, trans_eers, trans_r1s = q2_length_decay(
            trans_model, data, device, out_dir, lengths=length_steps,
            batch_size=args.batch_size, label="Transformer",
        )

        # Combined comparison charts
        q3_compare(
            lstm_model, trans_model, data, device, out_dir,
            lstm_train_losses, lstm_val_losses,
            trans_train_losses, trans_val_losses,
            max_len=args.max_len, batch_size=args.batch_size,
        )
        plot_q2_overlay(lengths, lstm_eers, lstm_r1s, trans_eers, trans_r1s, out_dir)

    print(f"\nAll charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
