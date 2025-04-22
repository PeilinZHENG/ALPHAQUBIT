# File: ai_models/alphaqubit.py
# This module implements the AlphaQubit decoder and training script.
# Data files live in the sibling "output" folder at project root, so we reference ../output/ by default.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class StabilizerEmbedder(nn.Module):
    """
    Embeds per-stabilizer input features via separate linear projections
    for each feature channel, then injects a learned index embedding.
    """
    def __init__(self, num_features: int, hidden_dim: int, num_stabilizers: int):
        super().__init__()
        self.feature_projs = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])
        self.index_embedding = nn.Embedding(num_stabilizers, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F = x.shape
        h = x.new_zeros(B, S, self.index_embedding.embedding_dim)
        for i, proj in enumerate(self.feature_projs):
            feat = x[..., i:i+1]
            h = h + proj(feat)
        idx = torch.arange(S, device=x.device)
        h = h + self.index_embedding(idx)
        return self.layer_norm(h)


class SyndromeTransformerLayer(nn.Module):
    """
    One recurrent step of the syndrome transformer.
    Supports non-square stabilizer counts by skipping spatial conv if mismatch.
    """
    def __init__(self, hidden_dim, num_heads, grid_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.grid_size = grid_size

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B, S, D = state.shape
        spatial = self.grid_size + 1
        # only apply conv if stabilizers form a square grid
        if spatial * spatial == S + 1:
            h = state.transpose(1,2).view(B, D, spatial, spatial)
            h = self.conv(h)
            h = h.view(B, D, S).transpose(1,2)
        else:
            h = state.new_zeros(B, S, D)
        state = self.norm1(state + h)
        attn_out, _ = self.attn(state, state, state)
        state = self.norm2(state + attn_out)
        ff_out = self.ff(state)
        state = self.norm3(state + ff_out)
        return state


class SyndromeTransformer(nn.Module):
    """
    Recurrent transformer core stacking multiple layers.
    """
    def __init__(self, hidden_dim, num_heads, num_layers, grid_size, dilation=1):
        super().__init__()
        self.layers = nn.ModuleList([
            SyndromeTransformerLayer(hidden_dim, num_heads, grid_size, dilation)
            for _ in range(num_layers)
        ])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            state = layer(state)
        return state


class ReadoutNetwork(nn.Module):
    """
    Aggregates final state to predict logical error.
    Uses spatial conv if stabilizers form a square grid, else global pooling.
    """
    def __init__(self, hidden_dim, grid_size, basis='X', num_stabilizers=None):
        super().__init__()
        self.grid_size = grid_size
        self.basis = basis
        self.num_stabilizers = num_stabilizers
        spatial = grid_size + 1
        # determine whether spatial conv can be applied
        if spatial >= 2 and num_stabilizers is not None and spatial * spatial == num_stabilizers + 1:
            self.use_conv = True
            kernel = 2
            self.scatter_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel)
            out_w = spatial - kernel + 1
            self.mlp = nn.Sequential(
                nn.Linear(out_w * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.use_conv = False
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            B, S, D = state.shape
            spatial = self.grid_size + 1
            h = state.transpose(1,2).view(B, D, spatial, spatial)
            h = self.scatter_conv(h)
            if self.basis == 'X':
                h = h.mean(dim=2)
            else:
                h = h.mean(dim=3)
            h = h.view(B, -1)
            return self.mlp(h).squeeze(-1)
        else:
            # global feature pooling
            h = state.mean(dim=1)  # (B, hidden_dim)
            return self.mlp(h).squeeze(-1)


class AlphaQubitDecoder(nn.Module):
    """Full recurrent-transformer decoder."""
    def __init__(self, num_features, hidden_dim, num_stabilizers,
                 grid_size, num_heads=4, num_layers=3, dilation=1, basis='X'):
        super().__init__()
        self.embedder = StabilizerEmbedder(num_features, hidden_dim, num_stabilizers)
        self.transformer = SyndromeTransformer(hidden_dim, num_heads,
                                               num_layers, grid_size, dilation)
        self.readout = ReadoutNetwork(hidden_dim, grid_size, basis,
                                      num_stabilizers=num_stabilizers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, R, S, F = inputs.shape
        state = inputs.new_zeros(B, S, self.embedder.index_embedding.embedding_dim)
        for r in range(R):
            emb = self.embedder(inputs[:, r])
            state = (state + emb) * (1 / math.sqrt(2))
            state = self.transformer(state)
        return self.readout(state)


# ----- Training Script -----

class PauliPlusDataset(Dataset):
    """
    PyTorch Dataset for Pauli+ syndrome sequences and logical labels.
    Expects two .npy files:
      - syndromes: shape (N, rounds, num_stabilizers, num_features)
      - logicals:  shape (N,) or (N,1)
    """
    def __init__(self, syndromes_path, logicals_path):
        x = np.load(syndromes_path)
        if x.ndim == 2:
            x = np.ascontiguousarray(x).reshape(x.shape[0], x.shape[1], 1, 1)
        elif x.ndim == 3:
            x = np.ascontiguousarray(x).reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        elif x.ndim != 4:
            raise ValueError(f"Unexpected syndromes array ndim: {x.ndim}")
        self.X = torch.tensor(x, dtype=torch.float32)
        y = np.load(logicals_path)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(
    model, train_loader, val_loader,
    epochs=10, lr=1e-3, device='cuda'
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        avg_train = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item() * Xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == yb).sum().item()
        avg_val = val_loss / len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)

        print(f"Epoch {ep}: Train Loss={avg_train:.4f},"
              f" Val Loss={avg_val:.4f}, Acc={acc:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--syndromes', default='./output/pauli_plus_syndromes_20250417_100710.npy')
    parser.add_argument('--logicals', default='./output/pauli_plus_logicals_20250417_100710.npy')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    dataset = PauliPlusDataset(args.syndromes, args.logicals)
    n = len(dataset)
    idx = torch.randperm(n)
    split = int(0.9 * n)
    train_ds = torch.utils.data.Subset(dataset, idx[:split])
    val_ds = torch.utils.data.Subset(dataset, idx[split:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    _, rounds, S, F = dataset.X.shape
    grid = int(math.sqrt(S + 1)) - 1  # may not be perfect square
    model = AlphaQubitDecoder(
        num_features=F,
        hidden_dim=128,
        num_stabilizers=S,
        grid_size=grid,
        num_heads=4,
        num_layers=3,
        dilation=1,
        basis='X'
    )
 
    model_path = os.path.abspath("alphaqubit_model.pth")
    if os.path.exists(model_path):
        print(f"Loading existing model from: {model_path}")
        model.load_state_dict(torch.load(model_path))

    train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    import os
    print("Saving model to:", os.path.abspath("alphaqubit_model.pth"))
    torch.save(model.state_dict(), os.path.abspath("alphaqubit_model.pth"))

 

