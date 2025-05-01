#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaQubit – Working MLA Implementation
"""

import os
import math
import argparse
from glob import glob
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

# ---------------------------------------------------------------------
#  Optimized Multi-Head Latent Attention
# ---------------------------------------------------------------------

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Rotary components
        self.rotary_dim = min(32, d_model // 4)
        self.rotary = nn.Linear(d_model, num_heads * self.rotary_dim)
        
        self._init_weights()

    def _init_weights(self):
        for lin in [self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.rotary]:
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.constant_(lin.bias, 0)

    def forward(self, x):
        B, S, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim)
        
        # Add rotary features
        r = self.rotary(x).view(B, S, self.num_heads, self.rotary_dim)
        q = torch.cat([q, r], dim=-1)
        k = torch.cat([k, r], dim=-1)
        
        # Attention computation
        scale = 1.0 / math.sqrt(self.head_dim + self.rotary_dim)
        q = q.transpose(1, 2) * scale  # [B, nh, S, hd+rd]
        k = k.transpose(1, 2)          # [B, nh, S, hd+rd]
        
        scores = torch.matmul(q, k.transpose(-2, -1))  # [B, nh, S, S]
        attn = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(attn, v.transpose(1, 2))  # [B, nh, S, hd]
        output = output.transpose(1, 2).reshape(B, S, -1)
        
        return self.out_proj(output)

# ---------------------------------------------------------------------
#  Stable AlphaQubit Components
# ---------------------------------------------------------------------

class StabilizerEmbedder(nn.Module):
    def __init__(self, num_features, hidden_dim, num_stabilizers):
        super().__init__()
        self.feature_projs = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])
        self.index_embed = nn.Embedding(num_stabilizers, hidden_dim)
        self.final_on = nn.Embedding(1, hidden_dim)
        self.final_off = nn.Embedding(1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Initialize properly
        for proj in self.feature_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0)
        nn.init.normal_(self.index_embed.weight, mean=0, std=0.02)

    def forward(self, x, final_mask):
        B, S, _ = x.shape
        h = torch.zeros(B, S, self.index_embed.embedding_dim, device=x.device)
        
        for i, proj in enumerate(self.feature_projs):
            h += proj(x[..., i:i+1])
            
        h += self.index_embed(torch.arange(S, device=x.device))
        
        # Final mask embeddings
        h += (final_mask == 1).unsqueeze(-1) * self.final_on(torch.tensor(0, device=x.device))
        h += (final_mask == 2).unsqueeze(-1) * self.final_off(torch.tensor(0, device=x.device))
        
        return self.norm(h)

class SyndromeTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_stabilizers, grid_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadLatentAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Gated FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        # Initialize FFN properly
        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.xavier_uniform_(self.ffn[2].weight)
        nn.init.constant_(self.ffn[0].bias, 0)
        nn.init.constant_(self.ffn[2].bias, 0)

    def forward(self, x, events, prev_events):
        # Attention block
        x = x + self.attn(self.norm1(x))
        
        # FFN block
        x = x + self.ffn(self.norm2(x))
        
        return x

class SyndromeTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, num_stabilizers, grid_size):
        super().__init__()
        self.layers = nn.ModuleList([
            SyndromeTransformerLayer(hidden_dim, num_heads, num_stabilizers, grid_size)
            for _ in range(num_layers)
        ])

    def forward(self, x, events, prev_events):
        for layer in self.layers:
            x = layer(x, events, prev_events)
        return x

class ReadoutNetwork(nn.Module):
    def __init__(self, hidden_dim, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, basis):
        B, S, D = x.shape
        d = self.grid_size
        
        # Add dummy stabilizer and reshape
        x = torch.cat([x.new_zeros(B, 1, D), x], dim=1)
        x = x.transpose(1, 2).view(B, D, d+1, d+1)
        
        # Convolution
        x = self.conv(x).permute(0, 2, 3, 1)  # [B, d, d, D]
        
        # Line-wise readout
        outputs = []
        for i in range(B):
            if basis[i] == 0:  # X basis
                lines = x[i].mean(dim=1)  # [d, D]
            else:  # Z basis
                lines = x[i].mean(dim=0)  # [d, D]
                
            logits = self.mlp(lines).squeeze()  # [d]
            outputs.append(logits.mean())
            
        return torch.stack(outputs)

class AlphaQubitDecoder(nn.Module):
    def __init__(self, num_features, hidden_dim, num_stabilizers, grid_size, num_heads=4, num_layers=3):
        super().__init__()
        self.embedder = StabilizerEmbedder(num_features, hidden_dim, num_stabilizers)
        self.transformer = SyndromeTransformer(hidden_dim, num_heads, num_layers, num_stabilizers, grid_size)
        self.readout = ReadoutNetwork(hidden_dim, grid_size)
        
        # Initialize output layer
        for layer in self.readout.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, inputs, basis, final_mask):
        B, R, S, F = inputs.shape
        
        # Initialize state
        state = torch.zeros(B, S, self.embedder.index_embed.embedding_dim, 
                           device=inputs.device)
        prev_events = torch.zeros(B, S, device=inputs.device)
        
        # Process each round
        for r in range(R):
            x = inputs[:, r]
            
            # Embed and combine
            emb = self.embedder(x, final_mask if r == R-1 else torch.zeros_like(final_mask))
            state = (state + emb) / math.sqrt(2.0)
            
            # Transformer
            state = self.transformer(state, x[..., 0], prev_events)
            prev_events = x[..., 0]
            
        return self.readout(state, basis)

class PauliPlusDataset(Dataset):
    """
    Loads syndromes/logicals and zero-pads dummy stabilisers so that
    S+1 is a perfect square (required by AlphaQubit geometry).
    Returns:
        features : (R, S_pad, F)
        basis_id : scalar long
        final_mask: (S_pad,)  (1 = on, 2 = off)
    """
    def __init__(self, synd_path: str, log_path: str, basis_id: int):
        y = np.load(log_path)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1)   # (N,)
        N = len(self.y)

        x = np.load(synd_path)                                   # raw syndromes

        # --- reshape to (N, R, S, C) ---------------------------
        if x.ndim == 2:                     # (N,S) or (R,S)
            if x.shape[0] == N:
                x = x[:, None, :, None]
            else:                           # single sample
                x = x[None, :, :, None]
        elif x.ndim == 3:                   # (N,R,S) or (R,S,C)
            if x.shape[0] != N:
                x = x[None, ...]            # single sample
            if x.ndim == 3:                 # no feature axis
                x = x[..., None]
        elif x.ndim != 4:
            raise ValueError(f"{synd_path}: unsupported shape {x.shape}")

        # --- append basis-ID channel ---------------------------
        basis_feat = np.full((*x.shape[:-1], 1), basis_id, dtype=x.dtype)
        x = np.concatenate([x, basis_feat], axis=-1)             # (N,R,S,F)

        # --- pad stabilisers -----------------------------------
        N, R, S, F = x.shape
        d = math.isqrt(S + 1)
        if d*d != S + 1:             # not a perfect square -> pad zeros
            d += 1
            pad = d*d - 1 - S
            x = np.concatenate([x, np.zeros((N, R, pad, F), dtype=x.dtype)],
                               axis=2)
            S += pad
        self.X = torch.tensor(x, dtype=torch.float32)            # (N,R,S,F)
        self.basis = torch.full((N,), basis_id, dtype=torch.long)

        # final-round on/off mask
        self.final_mask = torch.zeros(N, S, dtype=torch.long)
        for idx, (r, c) in enumerate([(i//d, i%d) for i in range(1, S+1)]):
            self.final_mask[:, idx] = 1 if (r+c)%2==0 else 2

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx], self.basis[idx], self.final_mask[idx]), self.y[idx]

# ---------------------------------------------------------------------
#  Helper functions & training loop
# ---------------------------------------------------------------------

def discover_all_pairs(out_dir: str) -> List[Tuple[str, str, int]]:
    pairs = []
    for basis, bid in [("x",0), ("z",1)]:
        for s in glob(os.path.join(out_dir, f"*syndromes_{basis}_*.npy")):
            l = s.replace("syndromes", "logicals")
            if os.path.exists(l):
                pairs.append((s, l, bid))
    return sorted(pairs)


def build_dataset(pairs) -> Dataset:
    datasets = [PauliPlusDataset(s,l,b) for s,l,b in pairs]
    return datasets[0] if len(datasets)==1 else ConcatDataset(datasets)

# ---------------------------------------------------------------------
#  Training Utilities (Modified for MLA)
# ---------------------------------------------------------------------

def train(model, tr_loader, va_loader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        steps_per_epoch=len(tr_loader),
        epochs=epochs
    )
    criterion = nn.BCEWithLogitsLoss()
    
    best_val = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}/{epochs}")
        for (xb, basis, mask), yb in pbar:
            xb, basis, mask, yb = xb.to(device), basis.to(device), mask.to(device), yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb, basis, mask)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for (xb, basis, mask), yb in va_loader:
                xb, basis, mask, yb = xb.to(device), basis.to(device), mask.to(device), yb.to(device)
                outputs = model(xb, basis, mask)
                val_loss += criterion(outputs, yb).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == yb).sum().item()
        
        avg_loss = total_loss / len(tr_loader)
        val_loss = val_loss / len(va_loader)
        val_acc = correct / len(va_loader.dataset)
        
        print(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "alphaqubit_mla_best.pth")
# ---------------------------------------------------------------------
#  Main Execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    pairs = []
    for basis, bid in [("x", 0), ("z", 1)]:
        for s in glob(os.path.join(args.output_dir, f"*syndromes_{basis}_*.npy")):
            l = s.replace("syndromes", "logicals")
            if os.path.exists(l):
                pairs.append((s, l, bid))
    
    if not pairs:
        raise RuntimeError(f"No data found in {args.output_dir}")
    
    dataset = ConcatDataset([PauliPlusDataset(s, l, b) for s, l, b in pairs])
    
    # Train/val split
    n = len(dataset)
    idx = torch.randperm(n)
    split = int(0.9 * n)
    
    tr_ds = torch.utils.data.Subset(dataset, idx[:split])
    va_ds = torch.utils.data.Subset(dataset, idx[split:])
    
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)

    # Model
    (x0, _, _), _ = dataset[0]
    R, S, F = x0.shape
    d = int(math.sqrt(S + 1))
    grid_size = d - 1
    
    print(f"Rounds={R} Stabilisers={S} Features={F} grid={d}×{d}")
    
    model = AlphaQubitDecoder(F, 128, S, grid_size)
    
    # Load checkpoint if available
    ckpt = "alphaqubit_mla.pth"
    if os.path.exists(ckpt):
        try:
            state_dict = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint {ckpt} (strict=False)")
        except Exception as e:
            print(f"Couldn't load checkpoint: {e}")
    
    # Train
    train(model, tr_loader, va_loader, args.epochs, args.lr, device)
    
    # Save final model
    torch.save(model.state_dict(), "alphaqubit_mla.pth")
    print("Training complete. Model saved to alphaqubit_mla.pth")