import os
import sys
# allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch_optimizer as optim_extra
from ai_models.model import AlphaQubitDecoder
import stim
import math

MODEL_FILENAME = "alphaqubit_model.pth"  # pretrained base model in project root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune AlphaQubit decoder on Google experiment data"
    )
    parser.add_argument(
        "--data-root", "-d", required=True,
        help="Path to the root directory containing experiment subfolders"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=1024,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", "-w", type=float, default=1e-3,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=30,
        help="Number of fine-tuning epochs"
    )
    parser.add_argument(
        "--train-samples", "-t", type=int, default=19880,
        help="Number of training samples"
    )
    parser.add_argument(
        "--valid-samples", "-v", type=int, default=5120,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--patience", "-p", type=int, default=5,
        help="Early stopping patience"
    )
    return parser.parse_args()


class SingleFolderDataset(Dataset):
    """Loads detection_events.b8 and obs_flips_actual.01 for one experiment folder."""
    def __init__(self, folder_path):
        # Load labels (observable flips)
        lbl_file = os.path.join(folder_path, "obs_flips_actual.01")
        if not os.path.exists(lbl_file):
            raise FileNotFoundError(f"Label file not found: {lbl_file}")
        lbls = np.loadtxt(lbl_file, dtype=int)
        n_shots = lbls.shape[0]

        # Load detection events bits
        det_file = os.path.join(folder_path, "detection_events.b8")
        if not os.path.exists(det_file):
            raise FileNotFoundError(f"Detection events not found: {det_file}")
        raw = np.fromfile(det_file, dtype=np.uint8)
        bits = np.unpackbits(raw)

        # Infer detectors count from bits per shot
        if bits.size % n_shots != 0:
            raise ValueError(
                f"Cannot evenly divide {bits.size} bits by {n_shots} shots"
            )
        detectors = bits.size // n_shots

        # Reshape into (shots, 1, detectors, 1)
        data = bits.reshape(n_shots, detectors)
        data = data[:, None, :, None]

        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(lbls).float().unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_and_save_for(folder, args, device):
    print(f"\n=== Fine-tuning on {folder} ===")
    try:
        ds = SingleFolderDataset(folder)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    total = len(ds)
    train_n = min(args.train_samples, total)
    valid_n = min(args.valid_samples, total - train_n)
    test_n  = total - train_n - valid_n
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(
        ds, [train_n, valid_n, test_n]
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # Initialize model
    _, R, S, F = ds.data.shape
    # disable spatial conv
    grid_size = 0
    model = AlphaQubitDecoder(
        num_features=F,
        hidden_dim=128,
        num_stabilizers=S,
        grid_size=grid_size,
        num_heads=4,
        num_layers=3,
        dilation=1,
        basis='X'
    )
    # Load pretrained weights (filter mismatches)
    raw_sd = torch.load(MODEL_FILENAME)
    sd = {k:v for k,v in raw_sd.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(sd, strict=False)
    model.to(device)

    optimizer = optim_extra.Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val = float('inf')
    wait = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = sum(criterion(model(x.to(device)), y.to(device)).item() * x.size(0)
                       for x, y in valid_loader) / len(valid_ds)
        print(f"Epoch {epoch}: Validation Loss = {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            out = f"alphaqubit_{os.path.basename(folder)}.pth"
            print(f"Saving best model → {out}")
            torch.save(model.state_dict(), out)
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping.")
                break

    # Test evaluation
    best_file = f"alphaqubit_{os.path.basename(folder)}.pth"
    model.load_state_dict(torch.load(best_file))
    model.eval()
    correct = total_test = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = (torch.sigmoid(model(x)) > 0.5).float()
            correct += (preds == y).sum().item()
            total_test += y.size(0)
    print(f"Test Accuracy on {folder}: {correct/total_test:.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subdirs = [os.path.join(args.data_root, d)
               for d in os.listdir(args.data_root)
               if os.path.isdir(os.path.join(args.data_root, d))]
    if not subdirs:
        raise RuntimeError("No experiment subfolders found!")
    for folder in subdirs:
        train_and_save_for(folder, args, device)

if __name__ == '__main__':
    main()
