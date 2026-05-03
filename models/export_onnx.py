import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lstm import TIMING_COLS


class EmbedderONNX(nn.Module):
    """
    Export-friendly version of KeystrokeEmbedder.
    Replaces pack_padded_sequence with explicit masking so ONNX export works.
    Weights are identical — only the forward pass changes.
    """
    def __init__(self, lstm1, lstm2, fc, hidden_size):
        super().__init__()
        self.lstm1 = lstm1
        self.lstm2 = lstm2
        self.fc = fc
        self.hidden_size = hidden_size

    def forward(self, x, lengths):
        # x: (1, seq_len, 3)   lengths: (1,)
        out1, _ = self.lstm1(x)                         # (1, seq_len, hidden*2)
        _, (hn2, _) = self.lstm2(out1)                  # hn2: (2, 1, hidden)

        # Concatenate forward and backward final hidden states — matches training exactly
        # hn2[0] = forward direction final state, hn2[1] = backward direction final state
        last = torch.cat((hn2[0], hn2[1]), dim=1)       # (1, hidden*2)

        out = self.fc(last)
        return F.normalize(out, p=2, dim=1)


def export(model_dir):
    model_dir = Path(model_dir)

    with open(model_dir / "norm_stats.json") as f:
        stats = json.load(f)

    state = torch.load(model_dir / "embedder.pt", map_location="cpu", weights_only=True)
    hidden_size = state["lstm1.weight_ih_l0"].shape[0] // 4
    embed_dim   = state["fc.weight"].shape[0]

    # Reconstruct LSTM layers from saved weights
    lstm1 = nn.LSTM(input_size=len(TIMING_COLS), hidden_size=hidden_size,
                    num_layers=1, batch_first=True, bidirectional=True)
    lstm2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size,
                    num_layers=1, batch_first=True, bidirectional=True)
    fc = nn.Linear(hidden_size * 2, embed_dim)

    lstm1.load_state_dict({k.removeprefix("lstm1."): v for k, v in state.items() if k.startswith("lstm1.")})
    lstm2.load_state_dict({k.removeprefix("lstm2."): v for k, v in state.items() if k.startswith("lstm2.")})
    fc.load_state_dict({k.removeprefix("fc."): v for k, v in state.items() if k.startswith("fc.")})

    model = EmbedderONNX(lstm1, lstm2, fc, hidden_size)
    model.eval()

    # Dummy input with a small length; seq_len is fully dynamic
    dummy_x = torch.zeros(1, 10, len(TIMING_COLS))
    dummy_lengths = torch.tensor([10])

    out_path = model_dir / "embedder.onnx"
    torch.onnx.export(
        model,
        (dummy_x, dummy_lengths),
        str(out_path),
        input_names=["keystrokes", "lengths"],
        output_names=["embedding"],
        dynamic_axes={
            "keystrokes": {1: "seq_len"},
            "lengths":    {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model to {out_path}")
    print(f"hidden_size={hidden_size}  embed_dim={embed_dim}")
    print(f"Norm stats: means={stats['means']}  stds={stats['stds']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="D:/kdnlp_model")
    args = parser.parse_args()
    export(args.model_dir)
