# AlphaQubit：量子纠错模拟与机器学习解码工具包

## 概述

AlphaQubit 提供从量子纠错仿真数据生成、模型训练到解码评估的完整流程。支持多种噪声模型，基于 PyTorch 实现，并附带可视化工具，帮助研究者快速开展量子误码校正实验。

## 功能特性

- **数据生成**：支持检测误差模型（DEM）、电路去极化噪声（SI1000）以及泄漏、串扰与软读出（Pauli+）等多种噪声类型
- **配置实验**：通过 `configs/` 目录下的 YAML 文件灵活调整实验参数
- **表面码仿真**：`simulator/` 中实现量子表面码仿真器
- **预训练模型**：内置 `alphaqubit_model.pth`，可直接进行解码
- **可视化工具**：包括 `npy_viewer.py`（.npy 数据查看）和 `plot_alphaqubit_results.py`（绘制性能曲线）
- **PyTorch 集成**：在 `ai_models/` 中提供训练与推理脚本

## 安装

```bash
git clone https://github.com/xuda1979/ALPHAQUBIT.git
cd ALPHAQUBIT
# 可选：创建并激活虚拟环境
python3.8 -m venv venv
source venv/bin/activate
# 安装依赖
pip install --upgrade pip
pip install numpy scipy stim pyyaml torch

# 如需在华为 Ascend NPU 上训练，请安装带有 `torch.npu` 的 PyTorch 发行版并根据官方文档完成驱动配置。
```

## 仓库结构

```plaintext
├── ai_models/                   # 模型训练与解码脚本
├── configs/                     # 实验配置 YAML 文件
├── google_experiment_data/      # Google Sycamore 实验数据
├── simulator/                   # 量子纠错仿真器
├── generate_data.py             # 训练数据生成脚本
├── npy_viewer.py                # .npy 数据查看工具
├── plot_alphaqubit_results.py   # 解码性能绘制脚本
├── alphaqubit_model.pth         # 预训练模型
└── README.md                    # 项目说明
```

## 使用方法

### 1. 生成数据

```bash
# 检测误差模型（DEM）
python generate_data.py --model dem --samples 10000

# 电路去极化噪声（SI1000）
python generate_data.py --model si1000 --samples 10000

# 泄漏、串扰与软读出（Pauli+）
python generate_data.py --model pauli_plus --samples 10000

# 与论文对齐的物理噪声模型
python generate_data.py --model paper_aligned --samples 10000
```

生成的数据默认保存在 `output/` 目录。

### 2. 查看数据

```bash
python npy_viewer.py output/dem_samples.npy
```

### 3. 训练模型

```bash
python ai_models/train.py --config configs/dem.yaml
```

可在相应的 YAML 文件中调整超参数与噪声设置。

### 4. 解码与评估

```bash
python ai_models/decode.py \
  --model path/to/alphaqubit_model.pth \
  --data output/dem_samples.npy
```

解码结果（如逻辑错误率）将存于 `results/` 目录。

### 5. 绘制性能

```bash
python plot_alphaqubit_results.py --input results/metrics.json
```

## 全流程示例：从噪声文件生成到 NPU 上的全规模训练

1. **生成噪声样本**  
   使用 `run_create_all_samples.py` 调用 `google_qec_simulator` 为 `simulated_data/` 下的每个电路生成噪声 `.npz` 文件：
   ```bash
   python run_create_all_samples.py
   ```

2. **在 NPU 上训练所有样本**  
   在安装了 Ascend PyTorch（支持 `torch.npu`）的环境中运行：
   ```bash
   python run_training_all.py --npu
   ```
   该脚本会遍历 `simulated_data/*.npz`，并对每个文件执行 `ai_models/model_mla.py` 训练；当检测到多块 NPU 时，会自动将任务分配到所有设备，实现全规模并行训练。

3. **解码与评估**  
   训练完成后，可继续使用前述的 `ai_models/decode.py` 及可视化脚本对模型性能进行评估与展示。

## 配置

编辑 `configs/` 下的 YAML 文件，可自定义电路布局、误差率、样本数量及训练参数。

## 预训练模型

使用内置的预训练模型，无需重新训练即可快速进行解码：

```bash
python ai_models/decode.py --model alphaqubit_model.pth --data output/si1000_samples.npy
```

## 许可证

本项目基于 MIT 协议开源。

## 引用

若在研究中使用本工具包，请引用：

```bibtex
@article{Xu2024AlphaQubit,
  title={Learning high-accuracy error decoding for quantum processors},
  author={Xu, David and Google Quantum AI},
  journal={Nature},
  year={2025},
}
```

