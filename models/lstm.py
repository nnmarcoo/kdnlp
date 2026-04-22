import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Same timing features as the baseline plus physical key distance
TIMING_COLS = ["iki_ms", "dwell_ms", "flight_ms", "key_dist"]


# The baseline averaged all bigrams per user, throwing away the order.
# Here we keep the raw sequence of bigrams in the order they were typed.
# Each sample is one session: a sequence of feature vectors, one per bigram.
class KeystrokeDataset(Dataset):
    def __init__(self, df, pid_to_label, max_len=128):
        # df: the full train or test dataframe
        # pid_to_label: maps participant_id -> integer class label
        # max_len: cap sequence length (pad shorter, truncate longer)
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        # Return (feature_sequence, sequence_length, label)
        # feature_sequence: tensor of shape (max_len, n_features)
        # sequence_length: actual length before padding (so the LSTM can ignore pads)
        # label: integer user id
        pass


# A bidirectional LSTM that reads the keystroke sequence in both directions.
#
# 1. Input: sequence of (iki, dwell, flight, key_dist) per bigram
# 2. LSTM reads the sequence step by step, building up a hidden state
#    that summarizes what it has seen so far. Bidirectional means a second
#    LSTM reads the sequence backwards, so each position has context from
#    both before and after.
# 3. We take the final hidden states from both directions and concatenate them.
#    This is a fixed-size vector that represents the entire typing session.
# 4. A linear layer maps that vector to a score for each enrolled user.
# 5. The predicted user is whichever score is highest.
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        # input_size: number of features per timestep (4: iki, dwell, flight, dist)
        # hidden_size: how much information the LSTM carries at each step
        # num_layers: stacking multiple LSTMs on top of each other for more capacity
        # num_classes: number of enrolled users to classify
        self.lstm = None  # TODO: nn.LSTM(...)
        self.dropout = None  # TODO: nn.Dropout(...)
        self.fc = None  # TODO: nn.Linear(...)

    def forward(self, x, lengths):
        # x: (batch, max_len, input_size) - the padded sequences
        # lengths: (batch,) - actual length of each sequence
        #
        # Steps:
        # 1. Pack the padded sequences so the LSTM skips pad tokens
        # 2. Run through the LSTM
        # 3. Extract the final hidden state from both directions
        # 4. Concatenate forward and backward hidden states
        # 5. Pass through dropout and the classifier layer
        pass


# Standard PyTorch training: for each batch, run the model forward,
# compute cross-entropy loss (how wrong the predictions are), then
# backpropagate to update the weights.
def train_one_epoch(model, loader, optimizer, criterion, device):
    pass


# Run the model on test data without updating weights.
# Compute rank-1 accuracy (top prediction is correct).
def evaluate(model, loader, device):
    pass


def run(data_dir, n_users, seed=42, epochs=20, batch_size=64,
        hidden_size=128, num_layers=2, lr=1e-3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the same preprocessed CSVs the baseline uses
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    # Sample a subset of users, same as baseline for fair comparison
    # TODO: filter participants, build pid_to_label mapping

    # Normalize the timing features across the training set
    # (similar idea to z-normalization in baseline, but done per-feature
    # across all bigrams rather than per-bigram)
    # TODO: compute mean/std from training data, apply to both train and test

    # Create datasets and dataloaders
    # TODO: KeystrokeDataset + DataLoader for train and test

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: instantiate BiLSTM, optimizer, loss function

    # Training loop
    for epoch in range(epochs):
        # TODO: train_one_epoch, then evaluate, print progress
        pass

    # Final evaluation
    # TODO: report rank-1 accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./processed")
    parser.add_argument("--n_users", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    run(args.data_dir, args.n_users, args.seed, args.epochs,
        args.batch_size, args.hidden_size, args.num_layers, args.lr)
