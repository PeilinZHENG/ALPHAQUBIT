# google_qec_simulator/experiment_simulator.py
# --------------------------------------------------------------------
"""
Monte-Carlo simulator for Google-style DEM data with soft I/Q channels.

Usage
-----
    python -m google_qec_simulator.experiment_simulator  <exp_root>  \
           [--shots 1000] [--out ./sim_soft_data.npz]

`<exp_root>` is the directory that contains many sub-folders; every
sub-folder that contains   *.stim  and  *.dem  files with the same stem
is sampled.  A single compressed .npz bundle is written.
"""

from pathlib import Path
import argparse
import numpy as np

from circuit_utils import sample_detectors_obs
from data_manager  import DataManager
from main          import reshape_detectors, rounds_dets_per_round

# --------------------------------------------------------------------
# Soft-I/Q helpers  (same 1-D model as before)
# --------------------------------------------------------------------
def iq_sample(state: int, snr: float, t: float) -> float:
    mu = state
    z  = np.random.normal(mu, 1 / np.sqrt(snr))
    if state >= 1:
        z -= t * np.random.exponential(1.0)
    return z


def pdfs(z: float, snr: float, t: float):
    p0 = np.exp(-snr * z * z)
    p1 = 0.5 * np.exp(-snr * (z - 1) ** 2) + 0.5 * np.exp(-snr * z * z) * np.exp(-z / t) * (z > 0)
    p2 = 0.5 * np.exp(-snr * (z - 2) ** 2) + 0.5 * np.exp(-snr * (z - 1) ** 2) * np.exp(-(z - 1) / (2 * t)) * (z > 1)
    return p0, p1, p2


def posteriors(z: float, snr: float, t: float, priors=(0.495, 0.495, 0.01)):
    w0, w1, w2 = priors
    p0, p1, p2 = pdfs(z, snr, t)
    norm = w0 * p0 + w1 * p1 + w2 * p2
    if norm == 0.0:
        return 0.5, 0.0
    return (w1 * p1) / norm, (w2 * p2) / norm   # post1 , post2


def soft_channels(num, snr=10.0, t=0.01, leak_p=0.00275):
    states = np.random.choice([0, 1, 2], size=num,
                              p=[1 - leak_p - 0.5, 0.5, leak_p])
    z_vals = np.vectorize(iq_sample)(states, snr, t)
    post1, post2 = np.vectorize(posteriors)(z_vals, snr, t)
    return post1.astype(np.float32), post2.astype(np.float32)

# --------------------------------------------------------------------
def simulate_experiment_root(
    exp_root: Path,
    out_npz: Path,
    shots: int,
    snr: float = 10.0,
    t: float   = 0.01,
) -> None:
    dm = DataManager()

    for stim_path in exp_root.rglob("*.stim"):
        dem_path = stim_path.with_suffix(".dem")
        if not dem_path.exists():
            continue  # skip unpaired files

        dets_flat, obs = sample_detectors_obs(stim_path, dem_path, shots=shots)
        dets = reshape_detectors(dets_flat, dem_path)        # (N,R,S,1)

        R, S = rounds_dets_per_round(dem_path)
        post1, post2 = soft_channels(shots * R * S, snr=snr, t=t)
        post1 = post1.reshape(shots, R, S, 1)
        post2 = post2.reshape(shots, R, S, 1)

        data  = np.concatenate([dets, post1, post2], axis=-1)  # (N,R,S,3)
        dm.store(data, obs)

        print(f"[OK] {stim_path.relative_to(exp_root)}  shots={shots}")

    if not dm._data_blocks:
        raise RuntimeError("No (*.stim, *.dem) pairs were found!")

    dm.save(out_npz)

# --------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_root", type=Path,
                    help="Root folder that contains many sub-folders each "
                         "with *.stim & *.dem files.")
    ap.add_argument("--shots", type=int, default=1000,
                    help="Monte-Carlo shots per circuit (default 1000).")
    ap.add_argument("--out", type=Path, default=Path("./sim_soft_data.npz"),
                    help="Output .npz filename (default ./sim_soft_data.npz)")
    args = ap.parse_args()

    simulate_experiment_root(
        args.exp_root,
        args.out,
        shots=args.shots,
    )
    print("DONE.  →", args.out.resolve())
