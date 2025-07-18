#!/usr/bin/env python3
"""
run_fine_tune_all.py

For each .pth in the project root, invoke ai_models/fine_tune.py in-process,
passing --dataset as the corresponding folder (data_dir/<pth_stem>) and
--model_path as the .pth file. No subprocesses are spawned.
"""

import sys
import runpy
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune for all .pth files in cwd via ai_models/fine_tune.py"
    )
    parser.add_argument(
        "--data_dir", "-d",
        required=True,
        type=Path,
        help="Directory containing one subfolder per .pth (named by pth stem)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of epochs to pass to fine_tune"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=16,
        help="Batch size to pass to fine_tune"
    )
    parser.add_argument("--npu", action="store_true", help="Use NPUs for training")
    args = parser.parse_args()

    project_root  = Path(__file__).resolve().parent
    ai_models_dir = project_root / "ai_models"
    ft_script     = ai_models_dir / "fine_tune.py"

    if not args.data_dir.is_dir():
        print(f"Error: data_dir '{args.data_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    if not ft_script.exists():
        print(f"Error: cannot find {ft_script}", file=sys.stderr)
        sys.exit(1)

    # Ensure imports inside fine_tune.py resolve correctly
    sys.path.insert(0, str(ai_models_dir))
    sys.path.insert(0, str(project_root))

    # Collect all .pth files in the project root
    pth_files = sorted(project_root.glob("*.pth"))
    if not pth_files:
        print("No .pth files found in project root.", file=sys.stderr)
        return

    for pth in pth_files:
        # Dataset folder is data_dir / <pth_basename_without_ext>
        ds_folder = args.data_dir / pth.stem
        if not ds_folder.is_dir():
            print(f"Warning: dataset folder not found, skipping: {ds_folder}", file=sys.stderr)
            continue

        print(f"\n=== Fine-tuning on dataset {ds_folder} with model {pth.name} ===")
        # Build argv as if calling:
        # python ai_models/fine_tune.py --dataset ds_folder --model_path pth --epochs X --batch-size Y
        sys.argv = [
            str(ft_script),
            "--dataset",    str(ds_folder),
            "--model_path", str(pth),
            "--epochs",     str(args.epochs),
            "--batch-size", str(args.batch_size),
        ]
        if args.npu:
            sys.argv.append("--npu")
        runpy.run_path(str(ft_script), run_name="__main__")

if __name__ == "__main__":
    main()
