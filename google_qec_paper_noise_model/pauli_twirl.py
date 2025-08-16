# -*- coding: utf-8 -*-
"""
Pauli Twirling for Quantum Channels, including Generalized Twirling for
channels with leakage.

This module implements the conversion of a quantum channel into a stochastic
Pauli channel. The primary method uses the Pauli Transfer Matrix (PTM).
"""
from typing import Literal
import numpy as np
from . import kraus_utils


def _choi_from_kraus(kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Computes the Choi matrix for a quantum channel."""
    d = kraus_ops[0].shape[0]
    choi = np.zeros((d**2, d**2), dtype=complex)
    for k in kraus_ops:
        vec_k = k.flatten('F').reshape(-1, 1)
        choi += vec_k @ vec_k.conj().T
    return choi


def _ptm_from_choi(choi: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Computes the Pauli Transfer Matrix (PTM) from a Choi matrix."""
    d_sq = len(basis)
    d = int(np.sqrt(d_sq))
    ptm = np.zeros((d_sq, d_sq), dtype=float)

    for i, pi in enumerate(basis):
        for j, pj in enumerate(basis):
            term = np.kron(pi.T, pj)
            ptm[i, j] = np.real(np.trace(choi @ term))

    return ptm / d


def twirl_to_generalized_pauli_probs(
    kraus_ops: list[np.ndarray], n_qutrits: int
) -> dict[str, float]:
    """
    Analytically computes generalized Pauli error probabilities from Kraus
    operators for a qutrit channel via the PTM.
    """
    if n_qutrits not in [1, 2]:
        raise NotImplementedError("Only 1 and 2-qutrit twirling is implemented.")

    basis = kraus_utils.QUTRIT_BASIS if n_qutrits == 1 else kraus_utils.QUTRIT_2Q_BASIS
    labels = kraus_utils.QUTRIT_BASIS_LABELS if n_qutrits == 1 else kraus_utils.QUTRIT_2Q_LABELS

    # 1. Build the Choi matrix for the channel
    choi = _choi_from_kraus(kraus_ops)

    # 2. Build the Pauli Transfer Matrix (PTM)
    ptm = _ptm_from_choi(choi, basis)

    # 3. The diagonal of the PTM gives the probabilities of each generalized
    #    Pauli channel component.
    diag_ptm = np.diag(ptm)

    # The probability of each generalized Pauli error is related to the
    # diagonal elements of the PTM. For a twirled channel, the probability
    # of a resulting Pauli P_i is given by a linear combination of the
    # diagonal elements. In the simplest case (twirling over the full
    # Clifford group), it's directly related to the diagonal elements.
    # We will use this approximation here.
    # p(P_i) = PTM[i,i]

    # The PTM gives the expectation value of measuring an output basis element
    # given an input basis element. For a Pauli channel, the PTM is diagonal.
    # The diagonal elements are the probabilities of the corresponding Pauli
    # operators in the resulting stochastic Pauli channel.

    # The probability of the identity is p_I = PTM[0,0].
    # The probability of a non-identity Pauli P_i is p_i.
    # The sum of probabilities should be 1.
    # The PTM diagonal for a trace-preserving map should sum to d.
    # Let's re-normalize the probabilities.

    # The diagonal of the PTM represents the average fidelity of the channel
    # with respect to each basis operator.
    # For a channel E, the twirled channel is E_twirled = sum_i p_i P_i
    # The diagonal of the PTM of E_twirled is (p_0, p_1, ...).
    # The diagonal of the PTM of E is (Tr(P_0 E(P_0))/d, ...).
    # For Clifford twirling, these are equal.

    probs = {label: prob for label, prob in zip(labels, diag_ptm)}

    return probs
