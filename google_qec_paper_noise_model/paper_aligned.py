"""
Paper-aligned noise model wrapper.

Implements the paper's method: compose physical channels, apply GPT per channel
 to obtain Pauli(+leakage) behavior compatible with a Pauli/Clifford simulator,
 and inject correlated parallel-CZ errors, readout/reset errors, plus per-tick idles.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from simulator.pauli_plus_simulator import PauliPlusSimulator
from .gpt import gpt_single_qubit, amp_phase_kraus

@dataclass
class PaperAlignedNoiseConfig:
    # Timing
    cycle_ns: float = 1100.0
    # Decoherence
    T1_us: float = 68.0
    Tphi_us: float = 89.0
    p_heat: float = 0.0  # passive heating to |2>, folded through GPT (see note)
    # Readout / reset
    p_readout: float = 0.003
    p_reset: float = 0.003
    # DQLR (acts on |0>,|1>,|2|; folded via GPT on computational subspace here)
    dqlr_matrix: Tuple[Tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.05, 0.90, 0.05),
    )
    # CZ related mechanisms
    p_cz_leak_11_to_02: float = 0.001    # folded via GPT
    p_cz_crosstalk_ZZ: float = 0.0005
    p_cz_swap_like: float = 0.0005       # approximated by (XX+YY)/2
    p_leak_transport_12_to_30: float = 0.0005  # folded via GPT
    # “Excess” residual errors
    p_1q_excess: float = 0.0005
    p_cz_excess: float = 0.0010
    p_idle_excess: float = 0.0005
    # Injection controls
    twirl_idles_each_tick: bool = False
    twirl_after_1q_gates: bool = True

class PaperAlignedNoiseModel:
    def __init__(self, config: Dict, basis: str = "z"):
        self.cfg = self._load_cfg(config)
        self.basis = basis.lower()
        if self.basis not in ("x", "z"):
            raise ValueError("basis must be 'x' or 'z'")
        # Build existing Pauli+ simulator/circuit then instrument with paper noise
        self.sim = PauliPlusSimulator(config, basis)
        self.sim.apply_paper_aligned_noise(config=self.cfg.__dict__)

    def _load_cfg(self, raw: Dict) -> PaperAlignedNoiseConfig:
        d = PaperAlignedNoiseConfig()
        for k, v in (raw or {}).items():
            if hasattr(d, k):
                setattr(d, k, v)
        return d

    def sample(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        sampler = self.sim.circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(num_samples, separate_observables=True)
        return syndromes, logicals
