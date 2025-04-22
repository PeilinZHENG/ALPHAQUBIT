# File: plot_alphaqubit_results.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from ai_models.model import AlphaQubitDecoder

# —— 配置 —— 
MODEL_PATH = "alphaqubit_model.pth"  # 你保存的最佳模型文件
DEVICE     = torch.device("cpu")  # 或者 "cuda" 如果有 GPU
# 这里用 Google Sycamore 测试数据下载解压后，自己根据格式写加载函数：
def load_round_data(round_num):
    """
    返回 (X, y)：
      - X: numpy array，shape=(N_shots, rounds, num_stabilizers, 1)
      - y: numpy array，shape=(N_shots,)
    你可以复用 SingleFolderDataset 的逻辑，或者直接 np.load
    """
    # TODO: 实现数据加载
    pass

# —— 模型加载 —— 
def load_model():
    # 构造一个和训练时一模一样的模型
    # 注意要填入当初训练时的 hidden_dim、num_layers、grid_size…
    model = AlphaQubitDecoder(
        num_features=1,
        hidden_dim=128,
        num_stabilizers=1023,  # 需替换成你的实验 stabilizer 数
        grid_size=31,
        num_heads=4,
        num_layers=3,
        dilation=1,
        basis='X'
    )
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    return model

# —— 计算 LER —— 
def compute_ler(model, rounds):
    ler = []
    with torch.no_grad():
        for r in rounds:
            X, y = load_round_data(r)
            X = torch.from_numpy(X).to(DEVICE)
            logits = model(X).view(-1)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            ler.append(np.mean(preds != y))
    return ler

# —— 绘图函数 —— 
def plot_ler_per_round(rounds, ler):
    vals = [1 - 2*e for e in ler]
    plt.plot(rounds, vals, '-o')
    plt.xlabel("Error-correction round")
    plt.ylabel("1 - 2 × Logical Error")
    plt.title("AlphaQubit Fine-tuned LER vs Round")
    plt.grid(True)
    plt.show()

def plot_ler_comparison(distances, pre, fine):
    x = np.arange(len(distances))
    w = 0.35
    plt.bar(x-w/2, pre,  width=w, label="Pretrained")
    plt.bar(x+w/2, fine, width=w, label="Fine-tuned")
    plt.xticks(x, [f"d={d}" for d in distances])
    plt.xlabel("Code distance")
    plt.ylabel("Logical Error Rate")
    plt.title("AlphaQubit LER: Pre vs Fine")
    plt.legend()
    plt.show()

# —— 主流程 —— 
if __name__ == "__main__":
    model = load_model()
    rounds = [1,3,5,7,9,11,13,15,17,19,21,23,25]
    ler_rounds = compute_ler(model, rounds)
    plot_ler_per_round(rounds, ler_rounds)

    distances = [3, 5]
    # 下面数组要自己算出或从论文里拷过来
    pretrained_ler = [0.02901, 0.02748]
    finetuned_ler  = [0.02901, 0.02748]
    plot_ler_comparison(distances, pretrained_ler, finetuned_ler)
