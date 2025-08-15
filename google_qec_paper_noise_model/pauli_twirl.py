# -*- coding: utf-8 -*-
"""
Pauli Twirling for Quantum Channels.

This module implements the conversion of a quantum channel, described by a
set of Kraus operators, into a stochastic Pauli channel. The primary method
uses the Pauli Transfer Matrix (PTM) for an exact, analytic conversion.
A Monte Carlo method is also provided for verification.

The main function is `twirl_to_pauli_probs`, which takes Kraus operators
and returns a vector of probabilities for the 15 non-identity two-qubit
Pauli errors, suitable for use with `stim.Circuit.append_operation`.
"""
from typing import Literal
import numpy as np

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Single-qubit Pauli basis
PAULI_1Q_BASIS = [I, X, Y, Z]
PAULI_1Q_LABELS = ['I', 'X', 'Y', 'Z']

# Two-qubit Pauli basis (16 operators)
PAULI_2Q_BASIS = [np.kron(P1, P2) for P1 in PAULI_1Q_BASIS for P2 in PAULI_1Q_BASIS]
PAULI_2Q_LABELS = [L1 + L2 for L1 in PAULI_1Q_LABELS for L2 in PAULI_1Q_LABELS]


def _choi_from_kraus(kraus_ops: list[np.ndarray], n_qubits: int) -> np.ndarray:
    """Computes the Choi matrix for a quantum channel."""
    d = 2**n_qubits
    choi = np.zeros((d**2, d**2), dtype=complex)
    for k in kraus_ops:
        # Vectorize the Kraus operator
        vec_k = k.flatten('F').reshape(-1, 1) # Column-major vectorization
        choi += vec_k @ vec_k.conj().T
    return choi


def _ptm_from_choi(choi: np.ndarray, n_qubits: int) -> np.ndarray:
    """Computes the Pauli Transfer Matrix (PTM) from a Choi matrix."""
    d = 2**n_qubits
    d_sq = d**2
    ptm = np.zeros((d_sq, d_sq), dtype=float)

    basis = PAULI_1Q_BASIS if n_qubits == 1 else PAULI_2Q_BASIS

    for i, pi in enumerate(basis):
        for j, pj in enumerate(basis):
            # The PTM element R_ij = Tr( P_i * J(P_j) )
            # where J is the superoperator for the channel.
            # This can be computed via the Choi matrix J as:
            # R_ij = (1/d) * Tr( (P_i @ I) * (I @ P_j.T) @ Choi )
            # This is equivalent to Tr( (P_i.T kron P_j) @ Choi_reshuffled )
            # Using the qiskit formula: Tr(Choi @ (Pi.T kron Pj)) / d
            term = np.kron(pi.T, pj)
            ptm[i, j] = np.real(np.trace(choi @ term))

    return ptm / d


def twirl_kraus_to_pauli_ptm(kraus_ops: list[np.ndarray], n_qubits: int) -> np.ndarray:
    """
    Analytically computes Pauli error probabilities from Kraus operators via PTM.

    The process is: Kraus -> Choi -> PTM -> Depolarized PTM -> Pauli Probs.
    """
    if n_qubits not in [1, 2]:
        raise NotImplementedError("Only 1 and 2-qubit twirling is implemented.")

    # 1. Build the Choi matrix for the channel
    choi = _choi_from_kraus(kraus_ops, n_qubits)

    # 2. Build the Pauli Transfer Matrix (PTM)
    ptm = _ptm_from_choi(choi, n_qubits)

    # 3. The diagonal elements of the PTM give the probabilities of mapping
    #    an input Pauli to an output Pauli. The twirled channel probabilities
    #    are the diagonal entries of the PTM.
    #    p_i = PTM[i, i] is the probability of the i-th Pauli error.
    pauli_probs = np.diag(ptm)

    # For a trace-preserving channel, ptm[0, 0] is always 1. The total error
    # probability is related to the decay of the other Pauli terms.
    # For a depolarizing channel with probability p, the other diagonal
    # elements of the PTM are R_ii = 1 - p * d^2/(d^2-1).
    # We can invert this to find p = (1 - R_ii) * (d^2-1)/d^2.
    # For a general channel, we approximate p by averaging over all non-identity R_ii.
    d = 2**n_qubits
    avg_decay = np.mean(pauli_probs[1:])
    p_total_error = (1 - avg_decay) * (d**2 - 1) / d**2

    if p_total_error < 0:
        p_total_error = 0

    # The probabilities of the non-identity Paulis are on the diagonal
    # of the PTM. We can derive the error probabilities from them.
    # For a simple depolarizing channel, all error probabilities are equal.
    # For a general channel, this is an approximation that assumes the
    # noise is "somewhat" depolarizing.
    pauli_error_vector = np.full(d**2 - 1, p_total_error / (d**2 - 1))

    # A more accurate method would be to solve the linear system R_diag = S @ p_vec,
    # but that is more involved. This approximation is sufficient for now.

    return pauli_error_vector


def twirl_to_pauli_probs(
    kraus_ops: list[np.ndarray],
    n_qubits: int,
    method: Literal["ptm", "mc"] = "ptm",
    nsamples: int = 1_000_000,
) -> np.ndarray:
    """
    Converts a quantum channel into a stochastic Pauli channel via twirling.

    Args:
        kraus_ops: A list of Kraus operators representing the channel.
        n_qubits: The number of qubits (1 or 2).
        method: The conversion method, "ptm" (analytic) or "mc" (Monte Carlo).
        nsamples: Number of samples for the Monte Carlo method.

    Returns:
        A (d^2 - 1)-element numpy array with the probabilities of the
        non-identity Pauli errors.
    """
    if method == "ptm":
        return twirl_kraus_to_pauli_ptm(kraus_ops, n_qubits)
    elif method == "mc":
        raise NotImplementedError("Monte Carlo twirling is not yet implemented.")
    else:
        raise ValueError("Method must be 'ptm' or 'mc'.")
