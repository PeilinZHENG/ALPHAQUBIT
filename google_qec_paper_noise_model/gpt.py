"""
Generalized Pauli Twirling (GPT) utilities.

Implements the per-channel twirling step described in the Supplementary Information
of "Quantum error correction below the surface code threshold":
> "After the circuit is dressed with all relevant error channels, we apply a
> Generalized Pauli Twirling Approximation to each noise channel. This converts
> each noise channel to a generalized Pauli channel that also includes leakage,
> thereby making it compatible with Clifford simulation methods." (SI, IV.A.1)
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import numpy as np

# Single-qubit Pauli matrices in computational basis
PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

def _kraus_to_choi(kraus_ops: Iterable[np.ndarray]) -> np.ndarray:
    """Return Choi matrix of a CPTP channel from its Kraus operators."""
    choi = None
    for K in kraus_ops:
        v = np.kron(K, np.eye(K.shape[0]))
        vv = v.reshape(-1, 1) @ v.conj().reshape(1, -1)  # vec(K) vec(K)†
        choi = vv if choi is None else choi + vv
    return choi

def _twirl_choi_1q(choi: np.ndarray) -> np.ndarray:
    """Pauli twirl a 1-qubit Choi matrix."""
    twirled = np.zeros_like(choi, dtype=complex)
    # P(E)(ρ) = (1/4) Σ_P P† E(P ρ P†) P  -> Choi mapping by conjugation
    for P in PAULI_1Q.values():
        U = np.kron(P.T, P.conj())
        twirled += U @ choi @ U.conj().T
    return twirled / 4.0

def _pauli_probs_from_twirled_choi_1q(choi: np.ndarray) -> Dict[str, float]:
    """
    Extract diagonal Pauli transfer probabilities from a 1-qubit *twirled* Choi matrix.
    Returns probabilities for I, X, Y, Z that sum to 1 (on the computational subspace).
    """
    # Pauli transfer matrix entry R_{ab} = Tr[ (Pauli_a \otimes Pauli_b^T) Choi ] / 2
    R = {}
    labels = ["I", "X", "Y", "Z"]
    for a in labels:
        for b in labels:
            A = np.kron(PAULI_1Q[a], PAULI_1Q[b].T)
            R[(a, b)] = np.real_if_close(np.trace(A @ choi) / 2.0).item()
    # After twirl, the channel is diagonal in the Pauli basis: probabilities = PTM row of I
    # p_I = (1 + R[Z,Z] + R[X,X] + R[Y,Y]) / 4  (standard relation)
    rxx, ryy, rzz = R[("X", "X")], R[("Y", "Y")], R[("Z", "Z")]
    pI = (1.0 + rxx + ryy + rzz) / 4.0
    pX = (1.0 + rxx - ryy - rzz) / 4.0
    pY = (1.0 - rxx + ryy - rzz) / 4.0
    pZ = (1.0 - rxx - ryy + rzz) / 4.0
    probs = {"I": float(np.clip(pI, 0.0, 1.0)),
             "X": float(np.clip(pX, 0.0, 1.0)),
             "Y": float(np.clip(pY, 0.0, 1.0)),
             "Z": float(np.clip(pZ, 0.0, 1.0))}
    # Renormalize to sum to 1 on the computational subspace.
    s = sum(probs.values())
    if s > 0:
        for k in probs:
            probs[k] /= s
    return probs

def gpt_single_qubit(kraus_ops: Iterable[np.ndarray]) -> Dict[str, float]:
    """
    Apply GPT (Pauli twirl) to a general 1-qubit channel given by Kraus ops.
    Returns a dict of Pauli probabilities on the computational subspace.
    Leakage should be modeled by a separate classical branch (see paper_aligned.py).
    """
    choi = _kraus_to_choi(list(kraus_ops))
    twirled = _twirl_choi_1q(choi)
    return _pauli_probs_from_twirled_choi_1q(twirled)
