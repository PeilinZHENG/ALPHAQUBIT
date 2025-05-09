"""
DataManager
===========

Aggregates blocks and writes a single `.npz` with keys

  data         – detectors + soft channels   (N , R , S , 3) float32
  observables  – logical-observable bits     (N , n_obs)     uint8
"""
from pathlib import Path
import numpy as np


class DataManager:
    # -----------------------------------------------------------------
    def __init__(self) -> None:
        self._data_blocks: list[np.ndarray] = []
        self._obs_blocks:  list[np.ndarray] = []

    # -----------------------------------------------------------------
    def store(self, data: np.ndarray, obs: np.ndarray) -> None:
        self._data_blocks.append(data)
        self._obs_blocks.append(obs)

    # -----------------------------------------------------------------
    def save(self, outfile: Path) -> None:
        data = np.concatenate(self._data_blocks, axis=0)
        obs  = np.concatenate(self._obs_blocks,  axis=0)

        outfile.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outfile, data=data, observables=obs)
        print(f"[DataManager] wrote {outfile}  "
              f"(shots={data.shape[0]}  rounds={data.shape[1]})")
