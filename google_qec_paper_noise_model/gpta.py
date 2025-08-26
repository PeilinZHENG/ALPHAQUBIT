from __future__ import annotations
import numpy as np
from typing import List, Tuple

# Utilities for Pauli basis
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
PAULI_1Q = [I, X, Y, Z]

def _kron(*ops: np.ndarray) -> np.ndarray:
    out = np.array([[1.0+0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

def _project_to_qubit(K: np.ndarray) -> np.ndarray:
    """Take top-left 2x2 block of a 3x3 (qutrit) Kraus op to act in qubit subspace."""
    if K.shape[0] == 2:
        return K
    return K[:2,:2]

def _apply_kraus(Ks: List[np.ndarray], A: np.ndarray) -> np.ndarray:
    return sum(K @ A @ K.conj().T for K in Ks)

def _ptm_diag_from_kraus_1q(Ks: List[np.ndarray]) -> np.ndarray:
    """
    Compute the diagonal of the 1-qubit Pauli transfer matrix T (size 4),
    where T_{ij} = 1/2 Tr( P_i E(P_j) ), P_0=I. For Pauli-twirled channel
    the off-diagonals vanish; we need diag entries [1, λx, λy, λz].
    """
    # Project to computational subspace if 3x3
    Ks2 = [ _project_to_qubit(K) for K in Ks ]
    lam = np.zeros(4, dtype=float)
    for idx, P in enumerate(PAULI_1Q):
        EP = _apply_kraus(Ks2, P)
        lam[idx] = 0.5 * np.real(np.trace(P.conj().T @ EP))
    # enforce exact λ0 = 1 for CPTP in PTM
    lam[0] = 1.0
    return lam

def _hadamard4() -> np.ndarray:
    """Walsh-Hadamard-like matrix for mapping eigenvalues -> Pauli probabilities."""
    H = np.array([
        [1,  1,  1,  1],
        [1,  1, -1, -1],
        [1, -1,  1, -1],
        [1, -1, -1,  1],
    ], dtype=float)
    return H

def _lam_to_probs_1q(lam: np.ndarray) -> np.ndarray:
    """
    For a 1-qubit Pauli channel, probabilities over {I,X,Y,Z} relate to eigenvalues
    {1, λx, λy, λz} via p = (1/4) H lam, with H given by _hadamard4().
    """
    H = _hadamard4()
    p = (H @ lam.reshape(4,1)).ravel() / 4.0
    # numeric cleanup
    p[p < 0] = 0.0
    s = p.sum()
    if s <= 0:
        p = np.array([1,0,0,0], dtype=float)
    else:
        p /= s
    return p

def twirl_to_pauli_channel(Ks: List[np.ndarray], n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Pauli Twirling Approximation:
      - Accept Kraus ops (optionally in 3x3 space for leakage); we project to qubit subspace
        before twirling, preserving any leakage probability as 'lost' population.
      - Return (probs, leak), where
         probs: ndarray of shape (4,) for 1q or (16,) for 2q (factorized from 1q eigens)
         leak:  estimated leakage probability mass (population that left {|0>,|1>}).
    For 2 qubits we approximate the twirl as a Kronecker of 1q twirls (sufficient for sampling
    in Pauli+; exact two-qubit twirl can be added later if needed).
    """
    if n_qubits == 1:
        lam = _ptm_diag_from_kraus_1q(Ks)
        probs = _lam_to_probs_1q(lam)
        # crude leakage estimate: how much |1> populations do not stay in subspace on |1><1|
        rho1 = np.array([[0,0],[0,1]], dtype=complex)
        rho1_3 = np.zeros((3,3), complex); rho1_3[1,1] = 1.0
        # Promote Ks to 3x3 if needed for leakage accounting
        Ks3 = []
        for K in Ks:
            if K.shape[0] == 2:
                K3 = np.zeros((3,3), complex)
                K3[:2,:2] = K
                K3[2,2] = 1.0
                Ks3.append(K3)
            else:
                Ks3.append(K)
        E_rho1 = sum(K @ rho1_3 @ K.conj().T for K in Ks3)
        leak = float(np.real(E_rho1[2,2]))
        return probs, leak
    elif n_qubits == 2:
        # Factorized approximation: get 1q twirls for each side if channels are separable.
        # If not separable, we still use identical marginal twirls on both qubits, which
        # is sufficient to feed a Pauli+ sampler.
        lam = _ptm_diag_from_kraus_1q(Ks)  # use same marginal as 1q
        p1 = _lam_to_probs_1q(lam)
        probs = np.kron(p1, p1)  # 16 entries in {I,X,Y,Z}⊗{I,X,Y,Z}
        # leakage heuristic: same as 1q, twice
        _, leak1 = twirl_to_pauli_channel(Ks, 1)
        leak = 1.0 - (1.0 - leak1)**2
        return probs, leak
    else:
        raise NotImplementedError("Only 1 or 2 qubits supported")

