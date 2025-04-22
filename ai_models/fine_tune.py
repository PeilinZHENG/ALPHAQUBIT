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
import gc
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType

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
    """按需加载单个 shot，避免一次性解包所有数据"""
    def __init__(self, folder_path):
        # 1) 加载标签
        lbl_file = os.path.join(folder_path, "obs_flips_actual.01")
        if not os.path.exists(lbl_file):
            raise FileNotFoundError(f"Label file not found: {lbl_file}")
        lbls = np.loadtxt(lbl_file, dtype=int)
        self.labels = torch.from_numpy(lbls).float()
        self.n_shots = lbls.shape[0]

        # 2) 映射原始二进制文件
        det_file = os.path.join(folder_path, "detection_events.b8")
        if not os.path.exists(det_file):
            raise FileNotFoundError(f"Detection events not found: {det_file}")
        raw = np.memmap(det_file, dtype=np.uint8, mode='r')

        # 3) 计算每个 shot 的 bit 数和对应的 byte 数
        total_bits = raw.size * 8
        if total_bits % self.n_shots != 0:
            raise ValueError(
                f"Cannot evenly divide {total_bits} bits by {self.n_shots} shots"
            )
        self.bits_per_shot = total_bits // self.n_shots
        if self.bits_per_shot % 8 != 0:
            raise ValueError("bits_per_shot must be divisible by 8")
        self.bytes_per_shot = self.bits_per_shot // 8

        # 4) 保存映射对象
        self.raw = raw

    def __len__(self):
        return self.n_shots

    def __getitem__(self, idx):
        # 1) 取出对应 shot 的原始 bytes
        start = idx * self.bytes_per_shot
        end   = start + self.bytes_per_shot
        raw_shot = self.raw[start:end]

        # 2) 解包为 bits，并 reshape
        bits = np.unpackbits(raw_shot)
        data = bits.reshape(1, self.bits_per_shot, 1)  # (1, detectors, 1)

        # 3) 转为 float tensor
        x = torch.from_numpy(data).float()
        y = self.labels[idx]
        return x, y


 
def train_and_save_for(folder, args, device):
    print(f"\n=== Fine-tuning on {folder} ===")
    try:
        ds = SingleFolderDataset(folder)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # 划分数据集
    total = len(ds)
    train_n = min(args.train_samples, total)
    valid_n = min(args.valid_samples, total - train_n)
    test_n  = total - train_n - valid_n
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(
        ds, [train_n, valid_n, test_n]
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, pin_memory=True)

    # 初始化模型
    S = ds.bits_per_shot      # detectors 数量
    F = 1                     # 每个 detector 对应一个特征维度
    model = AlphaQubitDecoder(
        num_features=F,
        hidden_dim=128,
        num_stabilizers=S,
        grid_size=0,          # 关闭空间卷积
        num_heads=4,
        num_layers=3,
        dilation=1,
        basis='X'
    )

    # —— 插入 LoRA Adapter —— 
    from peft import LoraConfig, get_peft_model, TaskType
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,                           
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["ff.0", "ff.2", "attn"]
    )
    # wrap and train only the adapters
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # should list only LoRA params


    # 加载预训练权重（过滤不匹配项）
    raw_sd = torch.load(MODEL_FILENAME, map_location='cpu')
    sd = {
        k: v for k, v in raw_sd.items()
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape
    }
    model.load_state_dict(sd, strict=False)
    model.to(device)

    # 优化器和损失
    optimizer = optim_extra.Lamb(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    # 训练 + 早停
    best_val = float('inf')
    wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", unit="batch"):

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
        
            logits = model.base_model(x)
            loss   = criterion(model.base_model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                logits    = model.base_model(x)
                batch_loss = criterion(logits, y)
                val_loss  += batch_loss.item() * x.size(0)
        val_loss /= len(valid_ds)
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

    # 测试评估
    best_file = f"alphaqubit_{os.path.basename(folder)}.pth"
    model.load_state_dict(torch.load(best_file, map_location=device))
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y  = x.to(device), y.to(device)
            logits = model.base_model(x)
            preds  = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total   += y.size(0)
    print(f"Test Accuracy on {folder}: {correct/total:.4f}")


    # 释放内存
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
