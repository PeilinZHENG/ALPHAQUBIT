"""
Circuit helpers
===============

* If a matching *.dem / *.dem.json exists use that
* Otherwise compile the DETECTOR sampler straight from the *.stim
"""

from pathlib import Path
import stim
import numpy as np


# ---------------------------------------------------------------------
def _locate_dem(stim_path: Path) -> Path | None:
    stem = stim_path.stem
    for p in stim_path.parent.iterdir():
        if p.is_file() and p.stem.startswith(stem) and ".dem" in p.suffixes:
            return p
    return None


# ---------------------------------------------------------------------
def create_sampler(stim_path: Path) -> stim.CompiledDetectorSampler:
    """
    Returns a compiled sampler; prefers external DEM but falls back to
    circuit.compile_detector_sampler().
    """
    dem_path = _locate_dem(stim_path)

    if dem_path is not None:
        dem = stim.DetectorErrorModel.from_file(dem_path)
        return dem.compile_sampler()

    # Fallback: compile directly from circuit text
    circuit = stim.Circuit.from_file(stim_path)
    return circuit.compile_detector_sampler()


# ---------------------------------------------------------------------
def sample_detectors_obs(stim_path: Path, shots: int):
    """
    Returns
    -------
    dets : (shots , N_det)   uint8
    obs  : (shots , N_obs)   uint8
    """
    sampler = create_sampler(stim_path)
    dets, obs = sampler.sample(
        shots=shots,
        separate_observables=True,
    )
    return dets.astype(np.uint8), obs.astype(np.uint8)
