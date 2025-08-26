import numpy as np
from typing import Iterable

def soft_xor(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Soft XOR for Bernoulli probabilities:
        XOR(p, q) = p + q - 2 p q
    Works element-wise for arrays.
    """
    return p + q - 2.0 * p * q

def soft_detection_sequence(p_meas_1: Iterable[np.ndarray]) -> np.ndarray:
    """
    Given a time-ordered iterable of soft probabilities P(m_t = 1),
    return soft detection-event probabilities for edges between rounds:
        e_t = XOR(m_t, m_{t-1}) in expectation.
    Output shape is (#rounds-1, ...).
    """
    probs = [np.asarray(x, dtype=float) for x in p_meas_1]
    outs = []
    for t in range(1, len(probs)):
        outs.append(soft_xor(probs[t-1], probs[t]))
    return np.stack(outs, axis=0)

