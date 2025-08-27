"""
Generalized Pauli Twirling (GPT) utilities.
Implements the per-channel twirling step described in the paper's Methods/SI:
compose physical channels, then apply GPT so each becomes a Pauli(+leakage)
channel compatible with Clifford/Pauli simulation.
"""
from __future__ import annotations
from typing import Dict, Iterable, List
import numpy as np

# 1-qubit Pauli matrices
PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

def amp_phase_kraus(dt_us: float, T1_us: float, Tphi_us: float) -> List[np.ndarray]:
    """Compose amplitude damping (T1) then dephasing (Tphi) over dt_us."""
    gamma = 1.0 - np.exp(-dt_us / max(T1_us, 1e-12))
    lam = 1.0 - np.exp(-dt_us / max(Tphi_us, 1e-12))
    # amplitude damping
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    # dephasing
    D0 = np.sqrt(1 - lam) * np.eye(2, dtype=complex)
    D1 = np.sqrt(lam) * np.array([[1, 0], [0, -1]], dtype=complex)
    return [D @ K for K in (K0, K1) for D in (D0, D1)]

def _kraus_to_choi(kraus_ops: Iterable[np.ndarray]) -> np.ndarray:
    choi = None
    for K in kraus_ops:
        v = np.kron(K, np.eye(K.shape[0]))
        vv = v.reshape(-1, 1) @ v.conj().reshape(1, -1)
        choi = vv if choi is None else choi + vv
    return choi

def _twirl_choi_1q(choi: np.ndarray) -> np.ndarray:
    """Pauli twirl a 1-qubit Choi matrix."""
    twirled = np.zeros_like(choi, dtype=complex)
    for P in PAULI_1Q.values():
        U = np.kron(P.T, P.conj())
        twirled += U @ choi @ U.conj().T
    return twirled / 4.0

def _pauli_probs_from_twirled_choi_1q(choi: np.ndarray) -> Dict[str, float]:
    labels = ["I", "X", "Y", "Z"]
    R = {}
    for a in labels:
        for b in labels:
            A = np.kron(PAULI_1Q[a], PAULI_1Q[b].T)
            R[(a, b)] = float(np.real(np.trace(A @ choi) / 2.0))
    rxx, ryy, rzz = R[("X", "X")], R[("Y", "Y")], R[("Z", "Z")]
    pI = (1.0 + rxx + ryy + rzz) / 4.0
    pX = (1.0 + rxx - ryy - rzz) / 4.0
    pY = (1.0 - rxx + ryy - rzz) / 4.0
    pZ = (1.0 - rxx - ryy + rzz) / 4.0
    probs = {"I": max(0.0, pI), "X": max(0.0, pX), "Y": max(0.0, pY), "Z": max(0.0, pZ)}
    s = sum(probs.values())
    if s > 0:
        for k in probs:
            probs[k] /= s
    return probs

def gpt_single_qubit(kraus_ops: Iterable[np.ndarray]) -> Dict[str, float]:
    """Apply GPT to a general 1-qubit channel given by Kraus ops."""
    choi = _kraus_to_choi(list(kraus_ops))
    twirled = _twirl_choi_1q(choi)
    return _pauli_probs_from_twirled_choi_1q(twirled)
