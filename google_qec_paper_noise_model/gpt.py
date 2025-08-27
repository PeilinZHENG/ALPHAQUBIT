"""Generalized Pauli Twirling helpers."""
from __future__ import annotations
from typing import List, Dict
import numpy as np

from .gpta import twirl_to_pauli_channel


def gpt_single_qubit(kraus_ops: List[np.ndarray]) -> Dict[str, float]:
    """Return Pauli probabilities of a single-qubit channel via GPT.

    Args:
        kraus_ops: Kraus operators representing the channel. Operators may
            either act on the qubit subspace (2x2) or include a leakage level
            (3x3); in the latter case population leaving the computational
            subspace is ignored in the returned probabilities.

    Returns:
        Mapping from Pauli operators ("I", "X", "Y", "Z") to their
        corresponding probabilities after generalized Pauli twirling.
    """
    probs, _ = twirl_to_pauli_channel(kraus_ops, 1)
    return {k: float(p) for k, p in zip(["I", "X", "Y", "Z"], probs)}
