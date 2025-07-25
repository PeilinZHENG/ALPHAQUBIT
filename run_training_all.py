#!/usr/bin/env python3
"""
run_training_all.py

Enhanced training launcher for ALPHAQUBIT.  It iterates over all `.npz`
files under ``simulated_data`` and trains a model on each using
``ai_models/model_mla.py``.  When running with the ``--npu`` flag and
multiple Ascend NPUs are available the launcher dispatches multiple
training processes concurrently, pinning each process to a single device
via the ``NPU_VISIBLE_DEVICES`` environment variable.  This allows
utilisation of all available devices.  If NPUs are not present or
the ``--npu`` flag is omitted the script falls back to the original
serial behaviour and uses GPUs or the CPU as appropriate.

Usage:
    python run_training_all.py [--npu]

"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import List

try:
    import torch  # type: ignore
except ImportError:
    # torch may not be available at import time; handle gracefully
    torch = None  # type: ignore

# Training parameters.  Strings are used here as these values are passed
# directly on the command line to the child process.
EPOCHS: str = "20"
BATCH_SIZE: str = "16"
SCRIPT: Path = Path("ai_models/model_mla.py")
DATA_DIR: Path = Path("simulated_data")


def main() -> None:
    """Entry point of the training launcher."""
    parser = argparse.ArgumentParser(
        description=(
            "Train all NPZ files serially or in parallel across NPUs/GPUs. "
            "When --npu is supplied and multiple NPUs are available the "
            "training tasks will be dispatched concurrently across devices."
        )
    )
    parser.add_argument(
        "--npu",
        action="store_true",
        help=(
            "Use NPUs for training (requires Ascend PyTorch with torch.npu support). "
            "If multiple devices are available the launcher will schedule jobs "
            "across them in parallel."
        ),
    )
    args = parser.parse_args()

    # Collect all npz files in the data directory
    npz_files: List[Path] = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {DATA_DIR}")
        return

    # Determine the number of devices available.  Prefer NPUs when --npu is
    # specified, otherwise fall back to CUDA.  When torch is unavailable we
    # assume a single CPU device.
    device_count = 1
    if args.npu and torch is not None and hasattr(torch, "npu"):
        try:
            device_count = getattr(torch.npu, "device_count", lambda: 1)()
        except Exception:
            device_count = 1
    elif not args.npu and torch is not None and torch.cuda.is_available():
        device_count = torch.cuda.device_count()

    # Helper to build the command list for a single training invocation
    def build_cmd(npz: Path) -> List[str]:
        cmd: List[str] = [
            "python",
            str(SCRIPT),
            "--epochs",
            EPOCHS,
            "--batch_size",
            BATCH_SIZE,
            "--npz_file",
            str(npz),
        ]
        if args.npu:
            cmd.append("--npu")
        return cmd

    # If more than one device is available we run training tasks in parallel.
    # Each spawned process is pinned to a single device using the appropriate
    # visibility environment variable.  The concurrency level is capped at
    # the number of devices to avoid oversubscription.
    if device_count > 1:
        print(
            f"Detected {device_count} {'NPUs' if args.npu else 'GPUs'}. "
            "Launching tasks in parallel."
        )
        processes: List[subprocess.Popen] = []
        for idx, npz in enumerate(npz_files):
            device_idx = idx % device_count
            env = os.environ.copy()
            cmd = build_cmd(npz)
            # Set device visibility for the child process
            if args.npu and torch is not None and hasattr(torch, "npu"):
                env["NPU_VISIBLE_DEVICES"] = str(device_idx)
            elif not args.npu and torch is not None and torch.cuda.is_available():
                env["CUDA_VISIBLE_DEVICES"] = str(device_idx)
            print(f"[async] starting on device {device_idx}: {' '.join(cmd)}")
            processes.append(subprocess.Popen(cmd, env=env))
            # Limit the number of concurrent processes to the number of devices
            if len(processes) >= device_count:
                finished = processes.pop(0)
                finished.wait()
        # Wait for any remaining processes to finish
        for p in processes:
            p.wait()
        return

    # Serial fallback: one training process at a time
    for npz in npz_files:
        cmd = build_cmd(npz)
        env = os.environ.copy()
        if args.npu and torch is not None and hasattr(torch, "npu"):
            # Pin to the first NPU for consistency
            env["NPU_VISIBLE_DEVICES"] = "0"
        elif not args.npu and torch is not None and torch.cuda.is_available():
            env["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
