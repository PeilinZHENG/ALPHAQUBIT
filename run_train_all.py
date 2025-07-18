#!/usr/bin/env python
"""
run_all_serial.py

Sequentially runs model_mla.py on every .npz file under ./simulated_data
without spawning concurrent processes.
"""

import subprocess
import argparse
from pathlib import Path

# Config
EPOCHS     = "20"
BATCH_SIZE = "16"
SCRIPT     = Path("ai_models/model_mla.py")
DATA_DIR   = Path("simulated_data")

def main():
    parser = argparse.ArgumentParser(description="Train all NPZ files serially")
    parser.add_argument("--npu", action="store_true", help="Use NPUs for training")
    args = parser.parse_args()
    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {DATA_DIR}")
        return

    for npz in npz_files:
        cmd = [
            "python", str(SCRIPT),
            "--epochs", EPOCHS,
            "--batch_size", BATCH_SIZE,
            "--npz_file", str(npz)
        ]
        if args.npu:
            cmd.append("--npu")
        print(f"\nRunning: {' '.join(cmd)}")
        # This will block until the script finishes for each file
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
