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
        self.sequences = []
        self.lengths = []
        self.labels = []
        self.max_len = max_len

        # Group by participant and session to build sequences
        grouped = df.groupby(["participant_id", "session_id"])
        
        for (pid, sid), group in grouped:
            if pid not in pid_to_label:
                continue
                
            # Get raw sequence of features
            seq = group[TIMING_COLS].values.astype(np.float32)
            length = len(seq)
            
            # Truncate if too long
            if length > max_len:
                seq = seq[:max_len]
                length = max_len
                
            # Pad with zeros if too short
            padded_seq = np.zeros((max_len, len(TIMING_COLS)), dtype=np.float32)
            padded_seq[:length, :] = seq
            
            self.sequences.append(padded_seq)
            self.lengths.append(length)
            self.labels.append(pid_to_label[pid])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return (feature_sequence, sequence_length, label)
        # feature_sequence: tensor of shape (max_len, n_features)
        # sequence_length: actual length before padding (so the LSTM can ignore pads)
        # label: integer user id
        x = torch.tensor(self.sequences[idx])
        length = torch.tensor(self.lengths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, length, label


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
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)        
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        # x: (batch, max_len, input_size) - the padded sequences
        # lengths: (batch,) - actual length of each sequence
        #
        # Steps:
        # 1. Pack the padded sequences so the LSTM skips pad tokens
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # 2. Run through the LSTM
        packed_out, (hn, cn) = self.lstm(packed_x)
        # 3. Extract the final hidden state from both directions
        hidden_fwd = hn[-2]
        hidden_bwd = hn[-1]
        # 4. Concatenate forward and backward hidden states
        out = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        # 5. Pass through dropout and the classifier layer
        out = self.dropout(out)
        out = self.fc(out)
        return out


# Standard PyTorch training: for each batch, run the model forward,
# compute cross-entropy loss (how wrong the predictions are), then
# backpropagate to update the weights.
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, lengths, labels in loader:
        x, labels = x.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x, lengths)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return total_loss / total, correct / total


# Run the model on test data without updating weights.
# Compute rank-1 accuracy (top prediction is correct).
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, lengths, labels in loader:
            x, labels = x.to(device), labels.to(device)
            outputs = model(x, lengths)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0


def run(data_dir, n_users, seed=42, epochs=20, batch_size=64,
        hidden_size=128, num_layers=2, lr=1e-3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading data...")

    # Load the same preprocessed CSVs the baseline uses
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    # Sample a subset of users, same as baseline for fair comparison
    # TODO: filter participants, build pid_to_label mapping
    unique_users = train_df['participant_id'].unique()
    sampled_users = unique_users[:n_users]
    pid_to_label = {pid: i for i, pid in enumerate(sampled_users)}
    train_df = train_df[train_df['participant_id'].isin(sampled_users)]
    test_df = test_df[test_df['participant_id'].isin(sampled_users)]
    
    # Normalize the timing features across the training set
    # (similar idea to z-normalization in baseline, but done per-feature
    # across all bigrams rather than per-bigram)
    # TODO: compute mean/std from training data, apply to both train and test
    print("Normalizing features...")
    means = train_df[TIMING_COLS].mean()
    stds = train_df[TIMING_COLS].std()
    
    train_df[TIMING_COLS] = (train_df[TIMING_COLS] - means) / stds
    test_df[TIMING_COLS] = (test_df[TIMING_COLS] - means) / stds

    # Create datasets and dataloaders
    # TODO: KeystrokeDataset + DataLoader for train and test
    print("Building sequences and Datasets...")
    train_dataset = KeystrokeDataset(train_df, pid_to_label)
    test_dataset = KeystrokeDataset(test_df, pid_to_label)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: instantiate BiLSTM, optimizer, loss function
    print(f"Training on {device}...")
    model = BiLSTM(
        input_size=len(TIMING_COLS), 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        num_classes=len(pid_to_label)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()    
    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


    # Final evaluation
    # TODO: report rank-1 accuracy
    print(f"Final Rank-1 Accuracy on Test Set: {test_acc:.4f}")


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
