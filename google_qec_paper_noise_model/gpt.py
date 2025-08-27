import numpy as np
from .gpta import twirl_to_pauli_channel


def amp_phase_kraus(dt_us: float, T1_us: float, Tphi_us: float):
    """Return Kraus ops approximating amplitude+phase damping for a time step."""
    p_amp = 1 - np.exp(-dt_us / T1_us) if T1_us > 0 else 0.0
    p_deph = 1 - np.exp(-dt_us / Tphi_us) if Tphi_us > 0 else 0.0
    # amplitude damping
    K0 = np.array([[1, 0], [0, np.sqrt(1 - p_amp)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p_amp)], [0, 0]], dtype=complex)
    # phase damping
    Kp0 = np.array([[1, 0], [0, np.sqrt(1 - p_deph)]], dtype=complex)
    Kp1 = np.array([[0, 0], [0, np.sqrt(p_deph)]], dtype=complex)
    Ks = [Kp0 @ K0, Kp0 @ K1, Kp1 @ K0, Kp1 @ K1]
    return Ks


def gpt_single_qubit(Ks):
    """Pauli-twirl the given Kraus ops and return a dict of Pauli error probs."""
    probs, _ = twirl_to_pauli_channel(Ks, 1)
    return {"I": probs[0], "X": probs[1], "Y": probs[2], "Z": probs[3]}
