from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np

from .gpt import gpt_single_qubit
from simulator.pauli_plus_simulator import PauliPlusSimulator


@dataclass
class PaperAlignedConfig:
    """Subset of noise parameters used by the paper-aligned model."""
    p_1q_excess: float = 0.0
    p_cz_excess: float = 0.0
    p_idle_excess: float = 0.0
    p_cz_leak_11_to_02: float = 0.0
    p_cz_crosstalk_ZZ: float = 0.0
    p_cz_swap_like: float = 0.0


class PaperAlignedNoiseModel:
    def __init__(self, config: Dict, basis: str = "z"):
        self.cfg = self._load_cfg(config)
        self.basis = basis.lower()
        if self.basis not in ("x", "z"):
            raise ValueError("basis must be 'x' or 'z'")
        # Build your existing Pauli+ circuit and then *instrument it* per paper
        self.sim = PauliPlusSimulator(config, basis)
        # The PauliPlusSimulator owns the Stim circuit; we attach paper-aligned noise.
        self.sim.apply_paper_aligned_noise(config=self.cfg.__dict__)

    def _load_cfg(self, config: Dict) -> PaperAlignedConfig:
        """Convert a raw dict of parameters into a structured config object."""
        fields = PaperAlignedConfig.__annotations__.keys()
        values = {k: float(config.get(k, 0.0)) for k in fields}
        return PaperAlignedConfig(**values)

    def sample(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample via Stim using the detector layout in the underlying circuit."""
        sampler = self.sim.circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(num_samples, separate_observables=True)
        return syndromes, logicals
