# Quantum Error Correction Simulator

Generates training data for machine-learning decoders using:
1. **DEM Model** (Detection Error Model)
2. **SI1000 Model** (Circuit depolarizing noise)
3. **Pauli+ Model** (Leakage + Cross-talk + Soft readouts)

## Requirements
- Python 3.8+
- `numpy`, `scipy`, `stim`, `pyyaml`

## Usage

### 1. Generate Data
```bash
# DEM Model
python generate_data.py --model dem --samples 10000

# SI1000 Model
python generate_data.py --model si1000 --samples 10000

# Pauli+ Model
python generate_data.py --model pauli_plus --samples 10000