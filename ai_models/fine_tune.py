# File: plot_alphaqubit_results.py

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from ai_models.model import AlphaQubitDecoder
from ai_models.fine_tune import SingleFolderDataset
import argparse

# —— 解析命令行参数 —— 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="Path to experiment subfolders")
    return parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— 模型加载 —— 
def load_model(model_path, folder_path):
    ds = SingleFolderDataset(folder_path)
    S = ds.bits_per_shot
    model = AlphaQubitDecoder(
        num_features=1,
        hidden_dim=128,
        num_stabilizers=S,
        grid_size=0,
        num_heads=4,
        num_layers=3,
        dilation=1,
        basis='X'
    )
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, ds

# —— 计算 LER —— 
def compute_ler(model, ds):
    loader = torch.utils.data.DataLoader(ds, batch_size=512)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x).view(-1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    ler = 1 - (correct / total)
    return ler

# —— 主流程 —— 
if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root

    folders = [os.path.join(data_root, d) for d in os.listdir(data_root)
               if os.path.isdir(os.path.join(data_root, d))]

    folder_names = []
    ler_values = []

    for folder in folders:
        folder_name = os.path.basename(folder)
        model_path = f"alphaqubit_{folder_name}.pth"  # 修改为从当前目录加载模型
        if not os.path.exists(model_path):
            print(f"[SKIP] No model for {folder_name}")
            continue
        print(f"[EVAL] {folder_name}")
        model, ds = load_model(model_path, folder)
        ler = compute_ler(model, ds)
        folder_names.append(folder_name)
        ler_values.append(ler)

    # —— 绘图 —— 
    x = np.arange(len(folder_names))
    plt.figure(figsize=(10,5))
    plt.bar(x, ler_values)
    plt.xticks(x, folder_names, rotation=90)
    plt.ylabel("Logical Error Rate (LER)")
    plt.title("LER per experiment folder (Fine-tuned models)")
    plt.tight_layout()
    plt.show()
