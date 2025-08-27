from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np

# Define Pauli matrices as numpy arrays for convenience
PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def amp_phase_kraus(dt_us: float, T1_us: float, Tphi_us: float) -> List[np.ndarray]:
    """Helper: compose amplitude damping(T1) then pure dephasing(Tphi) over dt_us.

    Args:
        dt_us: Duration of the noise in microseconds.
        T1_us: Amplitude damping time constant in microseconds.
        Tphi_us: Pure dephasing time constant in microseconds.

    Returns:
        List of Kraus operators describing the combined channel.
    """
    gamma = 1.0 - np.exp(-dt_us / max(T1_us, 1e-12))
    lam = 1.0 - np.exp(-dt_us / max(Tphi_us, 1e-12))
    # Amplitude damping Kraus operators
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    # Pure dephasing Kraus operators
    D0 = np.sqrt(1 - lam) * np.eye(2, dtype=complex)
    D1 = np.sqrt(lam) * np.array([[1, 0], [0, -1]], dtype=complex)
    # Compose: apply amplitude then dephasing
    return [D @ K for K in (K0, K1) for D in (D0, D1)]


def _kraus_to_choi(kraus_ops: Iterable[np.ndarray]) -> np.ndarray:
    """Compute the Choi matrix from a collection of Kraus operators."""
    choi = np.zeros((4, 4), dtype=complex)
    for K in kraus_ops:
        choi += np.kron(K, K.conj())
    return choi


def gpt_single_qubit(kraus_ops: Iterable[np.ndarray]) -> Dict[str, float]:
    """Return Pauli error probabilities for a single-qubit channel.

    Args:
        kraus_ops: Iterable of 2x2 Kraus operators describing the channel.

    Returns:
        Dictionary mapping 'I','X','Y','Z' to probabilities summing to 1.
    """
    choi = _kraus_to_choi(kraus_ops)
    probs: Dict[str, float] = {}
    for label, P in PAULI_1Q.items():
        v = np.kron(P, P.conj())
        probs[label] = float(np.real(np.trace(choi @ v)))
    s = sum(probs.values())
    if s > 0:
        for k in probs:
            probs[k] /= s
    return probs
