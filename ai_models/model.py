# import os
# import math
# import argparse
# from glob import glob
# from typing import List, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from tqdm import tqdm

# ###############################################
# # Model Components (Robust AlphaQubit-inspired)
# ###############################################

# class StabilizerEmbedder(nn.Module):
#     """Embeds per-stabilizer input features + learned index embedding."""
#     def __init__(self, num_features: int, hidden_dim: int, num_stabilizers: int):
#         super().__init__()
#         # Linear projection for each feature channel
#         self.feature_projs = nn.ModuleList([
#             nn.Linear(1, hidden_dim) for _ in range(num_features)
#         ])
#         # Learned index embedding for S positions
#         self.index_embedding = nn.Embedding(num_stabilizers, hidden_dim)
#         self.layer_norm = nn.LayerNorm(hidden_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, S, F]
#         B, S, F = x.shape
#         # sum feature projections
#         h = x.new_zeros(B, S, self.index_embedding.embedding_dim)
#         for i, proj in enumerate(self.feature_projs):
#             h = h + proj(x[..., i:i+1])
#         # add index embedding
#         idx = torch.arange(S, device=x.device)
#         h = h + self.index_embedding(idx)
#         return self.layer_norm(h)

# class SyndromeTransformerLayer(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int,
#         num_heads: int,
#         num_stabilizers: int,
#         grid_size: int,
#         use_attn_bias: bool = True,
#         use_dilated_convs: bool = True
#     ):
#         super().__init__()
#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads
#         self.grid_size = grid_size
#         # spatial inductive bias via convolutions
#         if use_dilated_convs:
#             self.convs = nn.ModuleList([
#                 nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=d, dilation=d)
#                 for d in (1, 2, 4)
#             ])
#         else:
#             self.convs = nn.ModuleList([nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)])
#         # attention projections
#         self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
#         self.o_proj = nn.Linear(hidden_dim, hidden_dim)
#         # optional learned attention bias
#         if use_attn_bias:
#             self.attn_bias = nn.Parameter(torch.zeros(num_heads, num_stabilizers, num_stabilizers))
#         else:
#             self.register_parameter('attn_bias', None)
#         # gated feed-forward
#         self.ff_proj = nn.Linear(hidden_dim, 4 * hidden_dim)
#         self.ff_gate = nn.Linear(hidden_dim, 4 * hidden_dim)
#         self.ff_out  = nn.Linear(4 * hidden_dim, hidden_dim)
#         # norms
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.norm3 = nn.LayerNorm(hidden_dim)

#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         # state: [B, S, D]
#         B, S, D = state.shape
#         # spatial conv branch if S+1 is square
#         spatial = self.grid_size + 1
#         if spatial * spatial == S + 1:
#             pad = state.new_zeros(B, 1, D)
#             h2d = torch.cat([pad, state], dim=1)
#             h2d = h2d.transpose(1, 2).view(B, D, spatial, spatial)
#             conv_sum = sum(conv(h2d) for conv in self.convs)
#             h_flat = conv_sum.view(B, D, -1).transpose(1, 2)
#             h = h_flat[:, 1:, :]
#         else:
#             h = torch.zeros_like(state)
#         state = self.norm1(state + h)
#         # self-attention
#         qkv = self.qkv_proj(state)
#         q, k, v = qkv.chunk(3, dim=-1)
#         q = q.reshape(B, self.num_heads, S, self.head_dim)
#         k = k.reshape(B, self.num_heads, S, self.head_dim)
#         v = v.reshape(B, self.num_heads, S, self.head_dim)
#         scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         if self.attn_bias is not None:
#             scores = scores + self.attn_bias.unsqueeze(0)
#         attn = torch.softmax(scores, dim=-1)
#         out = attn @ v  # [B, H, S, head_dim]
#         out = out.transpose(1, 2).reshape(B, S, D)
#         state = self.norm2(state + self.o_proj(out))
#         # gated feed-forward
#         proj = self.ff_proj(state)
#         gate = torch.sigmoid(self.ff_gate(state))
#         ff = self.ff_out(proj * gate)
#         state = self.norm3(state + ff)
#         return state

# class SyndromeTransformer(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int,
#         num_heads: int,
#         num_layers: int,
#         num_stabilizers: int,
#         grid_size: int,
#         use_attn_bias: bool = True,
#         use_dilated_convs: bool = True
#     ):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             SyndromeTransformerLayer(
#                 hidden_dim=hidden_dim,
#                 num_heads=num_heads,
#                 num_stabilizers=num_stabilizers,
#                 grid_size=grid_size,
#                 use_attn_bias=use_attn_bias,
#                 use_dilated_convs=use_dilated_convs
#             ) for _ in range(num_layers)
#         ])

#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         for layer in self.layers:
#             state = layer(state)
#         return state

# class ReadoutNetwork(nn.Module):
#     def __init__(self, hidden_dim: int):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         # state: [B, S, D]
#         return self.mlp(state.mean(dim=1)).squeeze(-1)

# class AlphaQubitDecoder(nn.Module):
#     def __init__(
#         self,
#         num_features: int,
#         hidden_dim: int,
#         num_stabilizers: int,
#         grid_size: int,
#         num_heads: int = 4,
#         num_layers: int = 3,
#         use_attn_bias: bool = True,
#         use_dilated_convs: bool = True
#     ):
#         super().__init__()
#         self.embedder = StabilizerEmbedder(
#             num_features=num_features,
#             hidden_dim=hidden_dim,
#             num_stabilizers=num_stabilizers
#         )
#         self.transformer = SyndromeTransformer(
#             hidden_dim=hidden_dim,
#             num_heads=num_heads,
#             num_layers=num_layers,
#             num_stabilizers=num_stabilizers,
#             grid_size=grid_size,
#             use_attn_bias=use_attn_bias,
#             use_dilated_convs=use_dilated_convs
#         )
#         self.readout = ReadoutNetwork(hidden_dim)

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         # inputs: [B, R, S, F]
#         B, R, S, F = inputs.shape
#         state = inputs.new_zeros(B, S, self.embedder.index_embedding.embedding_dim)
#         for r in range(R):
#             state = (state + self.embedder(inputs[:, r])) * (1 / math.sqrt(2))
#             state = self.transformer(state)
#         return self.readout(state)

# ###############################################
# # Dataset & utilities
# ###############################################
# class PauliPlusDataset(Dataset):
#     def __init__(self, syndromes_path: str, logicals_path: str, basis_id: int):
#         x = np.load(syndromes_path)
#         # reshape to [N, R, S, ?]
#         if x.ndim == 2:
#             x = x[:, :, None, None]
#         elif x.ndim == 3:
#             x = x[:, :, :, None]
#         if x.ndim != 4:
#             raise ValueError(f"Unexpected ndim {x.ndim}")
#         # append basis feature
#         basis_feat = np.full((*x.shape[:-1], 1), basis_id, dtype=x.dtype)
#         x = np.concatenate([x, basis_feat], axis=-1)
#         self.X = torch.tensor(x, dtype=torch.float32)
#         y = np.load(logicals_path)
#         self.y = torch.tensor(y, dtype=torch.float32).view(-1)

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]


# def discover_all_pairs(output_dir: str) -> List[Tuple[str, str, int]]:
#     pairs = []
#     for basis, bid in [("x", 0), ("z", 1)]:
#         for s_path in glob(os.path.join(output_dir, f"*syndromes_{basis}_*.npy")):
#             l_path = s_path.replace("syndromes", "logicals")
#             if os.path.exists(l_path):
#                 pairs.append((s_path, l_path, bid))
#             else:
#                 print(f"[Warn] missing logical for {s_path}")
#     return sorted(pairs)


# def build_dataset(pairs: List[Tuple[str, str, int]]) -> Dataset:
#     datasets = [PauliPlusDataset(s, l, bid) for s, l, bid in pairs]
#     return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


# def param_count(m: nn.Module) -> int:
#     return sum(p.numel() for p in m.parameters() if p.requires_grad)


# def choose_hidden_dim(num_features: int, num_stabilizers: int, grid_size: int, target: int = 50000) -> int:
#     for hd in range(4, 257, 4):
#         m = AlphaQubitDecoder(
#             num_features=num_features,
#             hidden_dim=hd,
#             num_stabilizers=num_stabilizers,
#             grid_size=grid_size
#         )
#         if abs(param_count(m) - target) / target < 0.05:
#             return hd
#     return 128

# ###############################################
# # Training & main
# ###############################################

# def train(model: nn.Module,
#           train_loader: DataLoader,
#           val_loader: DataLoader,
#           epochs: int,
#           lr: float,
#           device: str):
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.BCEWithLogitsLoss()
#     for ep in range(1, epochs + 1):
#         model.train()
#         total_loss = 0.0
#         for xb, yb in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
#             xb, yb = xb.to(device), yb.to(device)
#             logits = model(xb)
#             loss = loss_fn(logits, yb)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * xb.size(0)
#         train_loss = total_loss / len(train_loader.dataset)
#         print(f"Epoch {ep} train_loss={train_loss:.4f}")
#         # validation
#         model.eval()
#         val_loss = 0.0
#         correct = 0
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 logits = model(xb)
#                 val_loss += loss_fn(logits, yb).item() * xb.size(0)
#                 preds = (torch.sigmoid(logits) > 0.5).float()
#                 correct += (preds == yb).sum().item()
#         val_loss /= len(val_loader.dataset)
#         acc = correct / len(val_loader.dataset)
#         print(f"           val_loss={val_loss:.4f} acc={acc:.4f}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--output_dir",
#                         default=os.path.join(os.path.dirname(__file__), "..", "output"))
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--batch_size", type=int, default=128)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     args = parser.parse_args()
#     torch.manual_seed(42)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # discover data
#     pairs = discover_all_pairs(args.output_dir)
#     # filter to uniform S count
#     if len(pairs) > 1:
#         sample = np.load(pairs[0][0])
#         S0 = sample.shape[2] if sample.ndim >= 3 else 1
#         filtered = [p for p in pairs if (np.load(p[0]).shape[2] if np.load(p[0]).ndim >= 3 else 1) == S0]
#         if len(filtered) < len(pairs):
#             print(f"[Warn] mixed stabilizer counts; keeping {len(filtered)} files with S={S0}")
#         pairs = filtered
#     if not pairs:
#         raise RuntimeError(f"No data files found in {args.output_dir}")
#     dataset = build_dataset(pairs)
#     n = len(dataset)
#     # train/val split
#     idx = torch.randperm(n)
#     split = int(0.9 * n)
#     train_ds = torch.utils.data.Subset(dataset, idx[:split])
#     val_ds   = torch.utils.data.Subset(dataset, idx[split:])
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
#     val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
#     # infer dimensions
#     x0, _ = dataset[0]
#     _, R, S, F = x0.unsqueeze(0).shape
#     grid = int(round(math.sqrt(S + 1))) - 1
#     # choose hidden dim
#     hd = choose_hidden_dim(F, S, grid)
#     print(f"Chosen hidden_dim={hd}, params={param_count(AlphaQubitDecoder(F,hd,S,grid))}")
#     # build model
#     model = AlphaQubitDecoder(
#         num_features=F,
#         hidden_dim=hd,
#         num_stabilizers=S,
#         grid_size=grid,
#         num_heads=4,
#         num_layers=3,
#         use_attn_bias=True,
#         use_dilated_convs=True
#     )
#     ckpt = os.path.abspath(f"alphaqubit_{hd}.pth")
#     if os.path.exists(ckpt):
#         try:
#             model.load_state_dict(torch.load(ckpt, map_location="cpu"))
#             print(f"Loaded checkpoint from {ckpt}")
#         except Exception:
#             print("[Warn] checkpoint mismatch, training from scratch")
#     # train & save
#     train(model, train_loader, val_loader, args.epochs, args.lr, device)
#     torch.save(model.state_dict(), ckpt)
#     print(f"Saved model to {ckpt}")

# if __name__ == "__main__":
#     main()


import os
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
# Model Components (Robust AlphaQubit-inspired)
###############################################

class StabilizerEmbedder(nn.Module):
    """Embeds per-stabilizer input features and adds a learned index embedding.

    Aligns with 'Input representation' and 'StabilizerEmbedder' in Methods:
    - Projects each input channel (e.g., detection event, measurement, leakage prob) via a separate linear layer.
    - Adds a learned positional embedding per stabilizer index to break symmetry.
    - Applies LayerNorm as described.
    """
    def __init__(self, num_features: int, hidden_dim: int, num_stabilizers: int):
        super().__init__()
        # One linear projection per feature channel (F total)
        self.feature_projs = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])
        # Learned embedding for each stabilizer index (size = num_stabilizers)
        self.index_embedding = nn.Embedding(num_stabilizers, hidden_dim)
        # Normalize summed embeddings
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, S, F]
        B, S, F = x.shape
        # Sum projections across feature channels
        h = x.new_zeros(B, S, self.index_embedding.embedding_dim)
        for i, proj in enumerate(self.feature_projs):
            # Project the i-th feature slice and accumulate
            h = h + proj(x[..., i:i+1])
        # Add positional embedding for each stabilizer index i
        idx = torch.arange(S, device=x.device)
        h = h + self.index_embedding(idx)
        # LayerNorm to stabilize training
        return self.layer_norm(h)

class SyndromeTransformerLayer(nn.Module):
    """One layer of the syndrome transformer RNN block.

    Matches 'Syndrome transformer' in Methods:
    - Multi-scale dilated convolutions for local and longer-range spatial mixing.
    - Multi-head self-attention with optional learned attention bias embedding per stabilizer pair.
    - Gated feed-forward (GLU-style) with LayerNorms.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_stabilizers: int,
        grid_size: int,
        use_attn_bias: bool = True,
        use_dilated_convs: bool = True
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.grid_size = grid_size

        # Convolutions: kernel=3, dilations [1,2,4] for multi-scale spatial bias
        if use_dilated_convs:
            self.convs = nn.ModuleList([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=d, dilation=d)
                for d in (1, 2, 4)
            ])
        else:
            # fallback to single conv if disabled
            self.convs = nn.ModuleList([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            ])

        # QKV projection for self-attention
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        # Output projection back to hidden_dim
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # Optional learned attention bias: shape [heads, S, S]
        if use_attn_bias:
            self.attn_bias = nn.Parameter(torch.zeros(num_heads, num_stabilizers, num_stabilizers))
        else:
            self.register_parameter('attn_bias', None)

        # Gated feed-forward network (GLU variant)
        self.ff_proj = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.ff_gate = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.ff_out  = nn.Linear(4 * hidden_dim, hidden_dim)

        # Three LayerNorms for pre-/post-residuals
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: [B, S, D]
        B, S, D = state.shape

        # Spatial conv branch: pad with virtual 'no-stabilizer' and reshape to 2D grid
        spatial = self.grid_size + 1
        if spatial * spatial == S + 1:
            pad = state.new_zeros(B, 1, D)
            h2d = torch.cat([pad, state], dim=1)  # [B, S+1, D]
            h2d = h2d.transpose(1, 2).view(B, D, spatial, spatial)
            # sum dilated conv outputs
            conv_sum = sum(conv(h2d) for conv in self.convs)
            # flatten back, drop the padded index
            h_flat = conv_sum.view(B, D, -1).transpose(1, 2)
            h = h_flat[:, 1:, :]  # remove padding row
        else:
            # shape mismatch: skip conv branch
            h = torch.zeros_like(state)

        # First residual + norm
        state = self.norm1(state + h)

        # Self-attention
        qkv = self.qkv_proj(state)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to [B, heads, S, head_dim]
        q = q.reshape(B, self.num_heads, S, self.head_dim)
        k = k.reshape(B, self.num_heads, S, self.head_dim)
        v = v.reshape(B, self.num_heads, S, self.head_dim)
        # scaled dot-product
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.attn_bias is not None:
            scores = scores + self.attn_bias.unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # [B, heads, S, head_dim]
        # merge heads back
        out = out.transpose(1, 2).reshape(B, S, D)
        # second residual + norm
        state = self.norm2(state + self.o_proj(out))

        # Gated feed-forward block
        proj = self.ff_proj(state)
        gate = torch.sigmoid(self.ff_gate(state))
        ff = self.ff_out(proj * gate)
        # third residual + norm
        state = self.norm3(state + ff)
        return state

class SyndromeTransformer(nn.Module):
    """Stacks multiple SyndromeTransformerLayer blocks as the RNN core."""
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_stabilizers: int,
        grid_size: int,
        use_attn_bias: bool = True,
        use_dilated_convs: bool = True
    ):
        super().__init__()
        # Build num_layers blocks
        self.layers = nn.ModuleList([
            SyndromeTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_stabilizers=num_stabilizers,
                grid_size=grid_size,
                use_attn_bias=use_attn_bias,
                use_dilated_convs=use_dilated_convs
            ) for _ in range(num_layers)
        ])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            state = layer(state)
        return state

class ReadoutNetwork(nn.Module):
    """Aggregates per-stabilizer state into a single logit prediction."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Two-layer MLP: [hidden->hidden->1]
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Mean-pool over stabilizers then MLP
        return self.mlp(state.mean(dim=1)).squeeze(-1)

class AlphaQubitDecoder(nn.Module):
    """
    Full AlphaQubit architecture:
    - Embeds input features per stabilizer
    - Recurrently processes R rounds with SyndromeTransformer
    - ReadoutNetwork predicts logical error probability
    """
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_stabilizers: int,
        grid_size: int,
        num_heads: int = 4,
        num_layers: int = 3,
        use_attn_bias: bool = True,
        use_dilated_convs: bool = True
    ):
        super().__init__()
        self.embedder = StabilizerEmbedder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_stabilizers=num_stabilizers
        )
        self.transformer = SyndromeTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_stabilizers=num_stabilizers,
            grid_size=grid_size,
            use_attn_bias=use_attn_bias,
            use_dilated_convs=use_dilated_convs
        )
        self.readout = ReadoutNetwork(hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [B, R, S, F]
        B, R, S, F = inputs.shape
        # Initialize decoder state: one hidden vector per stabilizer
        state = inputs.new_zeros(B, S, self.embedder.index_embedding.embedding_dim)
        for r in range(R):
            # Embed current-round syndromes and update state
            x_r = inputs[:, r]  # [B, S, F]
            emb = self.embedder(x_r)
            # Residual recurrence scaling by 1/sqrt(2) to stabilize norm
            state = (state + emb) * (1 / math.sqrt(2))
            # Transformer block updates state based on all stabilizers
            state = self.transformer(state)
        # Final readout to logit
        return self.readout(state)

###############################################
# Dataset & utilities
###############################################
class PauliPlusDataset(Dataset):
    def __init__(self, syndromes_path: str, logicals_path: str, basis_id: int):
        x = np.load(syndromes_path)
        # reshape to [N, R, S, ?]
        if x.ndim == 2:
            x = x[:, :, None, None]
        elif x.ndim == 3:
            x = x[:, :, :, None]
        if x.ndim != 4:
            raise ValueError(f"Unexpected ndim {x.ndim}")
        # append basis feature
        basis_feat = np.full((*x.shape[:-1], 1), basis_id, dtype=x.dtype)
        x = np.concatenate([x, basis_feat], axis=-1)
        self.X = torch.tensor(x, dtype=torch.float32)
        y = np.load(logicals_path)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def discover_all_pairs(output_dir: str) -> List[Tuple[str, str, int]]:
    pairs = []
    for basis, bid in [("x", 0), ("z", 1)]:
        for s_path in glob(os.path.join(output_dir, f"*syndromes_{basis}_*.npy")):
            l_path = s_path.replace("syndromes", "logicals")
            if os.path.exists(l_path):
                pairs.append((s_path, l_path, bid))
            else:
                print(f"[Warn] missing logical for {s_path}")
    return sorted(pairs)


def build_dataset(pairs: List[Tuple[str, str, int]]) -> Dataset:
    datasets = [PauliPlusDataset(s, l, bid) for s, l, bid in pairs]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def choose_hidden_dim(num_features: int, num_stabilizers: int, grid_size: int, target: int = 50000) -> int:
    for hd in range(4, 257, 4):
        m = AlphaQubitDecoder(
            num_features=num_features,
            hidden_dim=hd,
            num_stabilizers=num_stabilizers,
            grid_size=grid_size
        )
        if abs(param_count(m) - target) / target < 0.05:
            return hd
    return 128

###############################################
# Training & main
###############################################

def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          epochs: int,
          lr: float,
          device: str):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {ep} train_loss={train_loss:.4f}")
        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += loss_fn(logits, yb).item() * xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)
        print(f"           val_loss={val_loss:.4f} acc={acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        default=os.path.join(os.path.dirname(__file__), "..", "output"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # discover data
    pairs = discover_all_pairs(args.output_dir)
    # filter to uniform S count
    if len(pairs) > 1:
        sample = np.load(pairs[0][0])
        S0 = sample.shape[2] if sample.ndim >= 3 else 1
        filtered = [p for p in pairs if (np.load(p[0]).shape[2] if np.load(p[0]).ndim >= 3 else 1) == S0]
        if len(filtered) < len(pairs):
            print(f"[Warn] mixed stabilizer counts; keeping {len(filtered)} files with S={S0}")
        pairs = filtered
    if not pairs:
        raise RuntimeError(f"No data files found in {args.output_dir}")
    dataset = build_dataset(pairs)
    n = len(dataset)
    # train/val split
    idx = torch.randperm(n)
    split = int(0.9 * n)
    train_ds = torch.utils.data.Subset(dataset, idx[:split])
    val_ds   = torch.utils.data.Subset(dataset, idx[split:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
    # infer dimensions
    x0, _ = dataset[0]
    _, R, S, F = x0.unsqueeze(0).shape
    grid = int(round(math.sqrt(S + 1))) - 1
    # choose hidden dim
    hd = choose_hidden_dim(F, S, grid)
    print(f"Chosen hidden_dim={hd}, params={param_count(AlphaQubitDecoder(F,hd,S,grid))}")
    # build model
    model = AlphaQubitDecoder(
        num_features=F,
        hidden_dim=hd,
        num_stabilizers=S,
        grid_size=grid,
        num_heads=4,
        num_layers=3,
        use_attn_bias=True,
        use_dilated_convs=True
    )
    ckpt = os.path.abspath(f"alphaqubit_{hd}.pth")
    if os.path.exists(ckpt):
        try:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"Loaded checkpoint from {ckpt}")
        except Exception:
            print("[Warn] checkpoint mismatch, training from scratch")
    # train & save
    train(model, train_loader, val_loader, args.epochs, args.lr, device)
    torch.save(model.state_dict(), ckpt)
    print(f"Saved model to {ckpt}")

if __name__ == "__main__":
    main()
