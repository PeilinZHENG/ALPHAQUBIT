import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch_optimizer as optim_extra
#from ai_models.model import AlphaQubitDecoder
 
from model_mla import AlphaQubitDecoder   
import stim
import math
import gc
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
import time

#MODEL_FILENAME = "alphaqubit_model.pth"  # pretrained base model in project root
MODEL_FILENAME = "alphaqubit_mla.pth" 


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
    def __init__(self, folder_path):



        # 1) Load labels
        lbl_file = os.path.join(folder_path, "obs_flips_actual.01")
        if not os.path.exists(lbl_file):
            raise FileNotFoundError(f"Label file not found: {lbl_file}")
        lbls = np.loadtxt(lbl_file, dtype=int)
        self.labels = torch.from_numpy(lbls).float()
        self.n_shots = lbls.shape[0]

        # 2) Map detection events
        det_file = os.path.join(folder_path, "detection_events.b8")
        if not os.path.exists(det_file):
            raise FileNotFoundError(f"Detection events not found: {det_file}")
        self.raw = np.memmap(det_file, dtype=np.uint8, mode='r')

        # 3) Calculate stabilizer dimensions
        total_bits = self.raw.size * 8  # Convert bytes to bits
        self.bits_per_shot = total_bits // self.n_shots
        
        # Validate alignment
        if total_bits % self.n_shots != 0:
            raise ValueError(f"Total bits {total_bits} not divisible by {self.n_shots} shots")
        if self.bits_per_shot % 8 != 0:
            raise ValueError("bits_per_shot must be divisible by 8")
        self.bytes_per_shot = self.bits_per_shot // 8

         
        # Calculate grid dimensions
        self.d = math.isqrt(self.bits_per_shot + 1)
        if self.d * self.d != self.bits_per_shot + 1:
            self.d += 1
        self.S_pad = self.d * self.d - 1
        
        # Should match what the model sees
        print(f"Stabilizers: Original={self.bits_per_shot}, Padded={self.S_pad}, Grid={self.d-1}x{self.d-1}")

        # 4) Determine basis from folder name
        basename = os.path.basename(folder_path)
        self.basis_id = 0 if "_bX_" in basename else 1

        # 5) Calculate padding for grid alignment
        d = math.isqrt(self.bits_per_shot + 1)
        if d * d != self.bits_per_shot + 1:
            d += 1  # Need to pad to next square
        self.S_pad = d * d - 1  # Total stabilizers after padding

        # 6) Create final_mask matching model_mla's logic
        self.final_mask = torch.zeros(self.S_pad, dtype=torch.long)
        for idx in range(self.S_pad):
            r, c = divmod(idx, d)
            # Alternate 1/2 pattern like PauliPlusDataset
            self.final_mask[idx] = 1 if (r + c) % 2 == 0 else 2

        # 7) Store original dimensions
        self.original_shape = (1, self.bits_per_shot, 1)  # (R, S, F)
    def __len__(self):
        return self.n_shots

    # In SingleFolderDataset.__getitem__():

    def __getitem__(self, idx):
        # 1. Load detection events (x)
        start = idx * self.bytes_per_shot
        end = start + self.bytes_per_shot
        raw_shot = self.raw[start:end]
        bits = np.unpackbits(raw_shot)
        
        # 2. Create x tensor [R=1, S_initial, F=1]
        x = torch.from_numpy(bits).float().view(1, -1, 1)  # (1, S_initial, 1)
        
        # 3. Pad to S_pad stabilizers
        if x.shape[1] < self.S_pad:
            x = torch.cat([
                x, 
                torch.zeros(1, self.S_pad - x.shape[1], 1)
            ], dim=1)
        
        # 4. Get label (y)
        y = self.labels[idx]
        
        # 5. Return formatted data
        return (x, 
                torch.tensor(self.basis_id),  # From __init__
                self.final_mask.clone()), y    # From __init__
 

def train_and_save_for(folder, args, device):
    print(f"\n=== Fine-tuning on {folder} ===")
    try:
        ds = SingleFolderDataset(folder)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Extract dimensions from dataset
    F = 1  # Single feature per stabilizer (detection event)
    S = ds.S_pad  # Padded stabilizer count from dataset

    # Split dataset
    total = len(ds)
    train_n = min(args.train_samples, total)
    valid_n = min(args.valid_samples, total - train_n)
    test_n = total - train_n - valid_n
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(
        ds, [train_n, valid_n, test_n]
    )

    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, pin_memory=True)

    # Initialize model
    # With proper grid calculation:
    d = int(math.sqrt(S + 1))  # S is the padded stabilizer count
    model = AlphaQubitDecoder(
        num_features=F,
        hidden_dim=128,
        num_stabilizers=S,
        grid_size=d-1,  # Proper grid calculation
        num_heads=4,
        num_layers=3
    )

    # Load pretrained weights if available
    ckpt_path = "alphaqubit_mla.pth"
    if os.path.exists(ckpt_path):
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {ckpt_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}\nStarting from scratch")
    else:
        print("No pretrained model found - initializing new model")

    model.to(device)

    # Initialize optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training loop with early stopping
    best_val = float('inf')
    wait = 0    
    last_checkpoint_time = time.time()
    checkpoint_interval = 30 * 60  # 30 minutes in seconds

    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for (x, basis, mask), y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            x, basis, mask, y = x.to(device), basis.to(device), mask.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x, basis, mask)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            # Check if 30 minutes have passed since last checkpoint
            current_time = time.time()
            if current_time - last_checkpoint_time >= checkpoint_interval:
                checkpoint_path = f"alphaqubit_{os.path.basename(folder)}_epoch{epoch}_interim.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"\nSaved interim checkpoint at {checkpoint_path}")
                last_checkpoint_time = current_time


        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x, basis, mask), y in valid_loader:
                x, basis, mask, y = x.to(device), basis.to(device), mask.to(device), y.to(device)
                outputs = model(x, basis, mask)
                val_loss += criterion(outputs, y).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), f"alphaqubit_{os.path.basename(folder)}.pth")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test evaluation
    model.load_state_dict(torch.load(f"alphaqubit_{os.path.basename(folder)}.pth", map_location=device))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, basis, mask), y in test_loader:
            x, basis, mask, y = x.to(device), basis.to(device), mask.to(device), y.to(device)
            outputs = model(x, basis, mask)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    print(f"Test Accuracy: {correct/total:.4f}")

    # Cleanup
    del model, optimizer, train_loader, valid_loader, test_loader, ds
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()



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
