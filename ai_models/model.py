import os
import re
import math
import argparse
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

###############################################
#  Model Components
###############################################

class StabilizerEmbedder(nn.Module):
    """Embeds per‑stabilizer input features + learned index embedding."""

    def __init__(self, num_features: int, hidden_dim: int, num_stabilizers: int):
        super().__init__()
        self.feature_projs = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(num_features)])
        self.index_embedding = nn.Embedding(num_stabilizers, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, S, F)
        B, S, F = x.shape
        h = x.new_zeros(B, S, self.index_embedding.embedding_dim)
        for i, proj in enumerate(self.feature_projs):
            h = h + proj(x[..., i : i + 1])
        idx = torch.arange(S, device=x.device)
        h = h + self.index_embedding(idx)
        return self.layer_norm(h)


class SyndromeTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, grid_size: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.grid_size = grid_size

    def forward(self, state: torch.Tensor) -> torch.Tensor:  # (B, S, D)
        B, S, D = state.shape
        spatial = self.grid_size + 1
        if spatial * spatial == S + 1:
            h = state.transpose(1, 2).view(B, D, spatial, spatial)
            h = self.conv(h).view(B, D, S).transpose(1, 2)
        else:
            h = torch.zeros_like(state)
        state = self.norm1(state + h)
        a, _ = self.attn(state, state, state)
        state = self.norm2(state + a)
        state = self.norm3(state + self.ff(state))
        return state


class SyndromeTransformer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, grid_size: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [SyndromeTransformerLayer(hidden_dim, num_heads, grid_size) for _ in range(num_layers)]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            state = layer(state)
        return state


class ReadoutNetwork(nn.Module):
    """Global average pooling over stabilizers followed by MLP."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:  # (B, S, D)
        return self.mlp(state.mean(dim=1)).squeeze(-1)


class AlphaQubitDecoder(nn.Module):
    """Full model. Basis info is embedded as an extra feature channel."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_stabilizers: int,
        grid_size: int,
        num_heads: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        self.embedder = StabilizerEmbedder(num_features, hidden_dim, num_stabilizers)
        self.transformer = SyndromeTransformer(hidden_dim, num_heads, num_layers, grid_size)
        self.readout = ReadoutNetwork(hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # (B, R, S, F)
        B, R, S, F = inputs.shape
        state = inputs.new_zeros(B, S, self.embedder.index_embedding.embedding_dim)
        for r in range(R):
            state = (state + self.embedder(inputs[:, r])) * (1 / math.sqrt(2))
            state = self.transformer(state)
        return self.readout(state)

###############################################
#  Dataset utilities
###############################################

class PauliPlusDataset(Dataset):
    """Wraps (syndrome, logical, basis_id) files. Adds basis as extra feature."""

    def __init__(self, syndromes_path: str, logicals_path: str, basis_id: int):
        x = np.load(syndromes_path)
        if x.ndim == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        elif x.ndim == 3:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        elif x.ndim != 4:
            raise ValueError(f"Unexpected syndromes array ndim={x.ndim}")
        # add basis channel
        basis_feat = np.full((*x.shape[:-1], 1), basis_id, dtype=x.dtype)
        x = np.concatenate([x, basis_feat], axis=-1)
        self.X = torch.tensor(x, dtype=torch.float32)
        y = np.load(logicals_path)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

###############################################
#  Helper functions
###############################################

def discover_all_pairs(output_dir: str) -> List[Tuple[str, str, int]]:
    """Return list of (syndrome_path, logical_path, basis_id)."""
    pairs: List[Tuple[str, str, int]] = []
    for basis, bid in [("x", 0), ("z", 1)]:
        pattern = os.path.join(output_dir, f"*syndromes_{basis}_*.npy")
        for s_path in glob(pattern):
            l_path = re.sub(r"syndromes", "logicals", s_path, count=1)
            if os.path.exists(l_path):
                pairs.append((s_path, l_path, bid))
            else:
                print(f"[Warn] Missing logical for {os.path.basename(s_path)}")
    return sorted(pairs)


def build_dataset(pairs: List[Tuple[str, str, int]]) -> Dataset:
    datasets = [PauliPlusDataset(s, l, bid) for s, l, bid in pairs]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

###############################################
#  Model size control
###############################################

def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def choose_hidden_dim(num_features: int, num_stabilizers: int, grid_size: int, target: int = 50000) -> int:
    for hd in range(4, 257, 4):
        model = AlphaQubitDecoder(num_features, hd, num_stabilizers, grid_size)
        if abs(param_count(model) - target) / target < 0.05:
            return hd
    return 128  # fallback

###############################################
#  Training routine
###############################################

def train(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(1, epochs + 1):
        model.train(); total = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        print(f"Epoch {ep} train loss {total / len(train_loader.dataset):.4f}")
        # simple val
        model.eval(); vloss = 0; correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vloss += loss_fn(logits, yb).item() * xb.size(0)
                correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
        print(f"  val loss {vloss/len(val_loader.dataset):.4f} acc {correct/len(val_loader.dataset):.4f}")

###############################################
#  Main
###############################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = discover_all_pairs(args.output_dir)
    if not pairs:
        raise RuntimeError("No data files discovered in output_dir")

    dataset = build_dataset(pairs)
    n = len(dataset); idx = torch.randperm(n)
    split = int(0.9 * n)
    train_ds = torch.utils.data.Subset(dataset, idx[:split])
    val_ds = torch.utils.data.Subset(dataset, idx[split:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # derive dims from a sample
    x_sample, _ = dataset[0]
    _, rounds, S, F = x_sample.unsqueeze(0).shape
    grid = int(math.sqrt(S + 1)) - 1

    hidden_dim = choose_hidden_dim(F, S, grid)
    print(f"Chosen hidden_dim = {hidden_dim}")
    model = AlphaQubitDecoder(F, hidden_dim, S, grid)
    print(f"Trainable params: {param_count(model)}")

    # use hidden_dim in filename so checkpoints can coexist without shape clashes
    ckpt = os.path.abspath(f"alphaqubit_model_{hidden_dim}.pth")
    if os.path.exists(ckpt):
        try:
            print(f"Loading existing model from {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        except RuntimeError as e:
            print("[Warn] Checkpoint incompatible with current model (shape mismatch). "
                  "Starting from scratch. Delete old checkpoint if unintended.")
    
    train(model, train_loader, val_loader, args.epochs, args.lr, device)
    torch.save(model.state_dict(), ckpt)
    print(f"Saved model to {ckpt}")
    print(f"Saved model to {ckpt}")


if __name__ == "__main__":
    main()
