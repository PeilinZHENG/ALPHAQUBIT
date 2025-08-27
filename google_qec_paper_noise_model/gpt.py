from __future__ import annotations
import math
from typing import Dict, List
import numpy as np

from .channels import amplitude_damping_kraus, dephasing_kraus
from .gpta import twirl_to_pauli_channel


def amp_phase_kraus(*, dt_us: float, T1_us: float, Tphi_us: float) -> List[np.ndarray]:
    """Return Kraus ops for amplitude+phase damping over a time step."""
    # amplitude damping strength parameter (tau)
    tau = dt_us / T1_us
    K_amp = amplitude_damping_kraus(tau)
    # pure dephasing probability mapped to dephasing channel parameter
    p_phase = 0.5 * (1 - math.exp(-dt_us / Tphi_us))
    K_phase = dephasing_kraus(p_phase)
    # Compose amplitude then phase: K = P * A
    Ks: List[np.ndarray] = []
    for A in K_amp:
        for P in K_phase:
            Ks.append(P @ A)
    return Ks


def gpt_single_qubit(Ks: List[np.ndarray]) -> Dict[str, float]:
    """Twirl a 1-qubit channel to Pauli probabilities."""
    probs, _ = twirl_to_pauli_channel(Ks, 1)
    return {"I": float(probs[0]), "X": float(probs[1]), "Y": float(probs[2]), "Z": float(probs[3])}

