# -*- coding: utf-8 -*-
"""
A library of common quantum channels (Kraus operators).

This module provides functions to generate Kraus operators for various
noise channels relevant to superconducting qubits. These channels can be
used to construct detailed noise models for quantum circuits.
"""

import numpy as np
from scipy.linalg import sqrtm

# Pauli matrices (2x2 for qubits)
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Qutrit operators (3x3 for handling leakage to |2>)
QUTRIT_I = np.eye(3, dtype=complex)
# Generalized Gell-Mann matrices can be used, but for simple transitions,
# projection operators are more direct.
P0 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
P1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex)
P2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=complex)

# Transition operators
P10 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=complex) # |0><1|
P01 = P10.T.conj()                                             # |1><0|
P21 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=complex) # |1><2|
P12 = P21.T.conj()                                             # |2><1|


def kraus_amplitude_damping(time: float, t1: float) -> list[np.ndarray]:
    """
    Generates Kraus operators for an amplitude damping channel (T1 decay).

    Args:
        time: The duration of the noise channel.
        t1: The T1 relaxation time constant.

    Returns:
        A list of Kraus operators [E0, E1].
    """
    if t1 <= 0:
        return [PAULI_I]
    p = 1 - np.exp(-time / t1)
    e0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    e1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    return [e0, e1]


def kraus_dephasing(time: float, t2: float, t1: float) -> list[np.ndarray]:
    """
    Generates Kraus operators for a pure dephasing channel (T_phi).

    The T_phi decay rate is calculated from T1 and T2 as:
    1/T_phi = 1/T2 - 1/(2*T1)

    Args:
        time: The duration of the noise channel.
        t2: The T2 coherence time.
        t1: The T1 relaxation time.

    Returns:
        A list of Kraus operators [E0, E1].
    """
    if t2 <= 0:
        return [PAULI_I]

    # Ensure T2 is not longer than the theoretical limit
    if t2 > 2 * t1:
        # In this case, dephasing is negligible or parameters are unphysical
        t_phi_inv = 0
    else:
        t_phi_inv = 1/t2 - 1/(2*t1)

    if t_phi_inv <= 0:
        return [PAULI_I]

    t_phi = 1 / t_phi_inv
    p = 1 - np.exp(-2 * time / t_phi) # Note the factor of 2 for T_phi

    e0 = np.sqrt(1 - p/2) * PAULI_I
    e1 = np.sqrt(p/2) * PAULI_Z
    return [e0, e1]


def kraus_depolarizing(prob: float, n_qubits: int = 1) -> list[np.ndarray]:
    """
    Generates Kraus operators for a depolarizing channel.

    Args:
        prob: The probability of a Pauli error occurring.
        n_qubits: The number of qubits (1 or 2).

    Returns:
        A list of Kraus operators.
    """
    if n_qubits == 1:
        paulis = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        d = 4
    elif n_qubits == 2:
        paulis = [
            np.kron(P1, P2) for P1 in [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
            for P2 in [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        ]
        d = 16
    else:
        raise ValueError("Only 1 or 2 qubits are supported.")

    kraus_ops = [np.sqrt(prob / (d - 1)) * P for P in paulis[1:]]
    kraus_ops.insert(0, np.sqrt(1 - prob) * paulis[0])
    return kraus_ops


# Ideal gate matrices
IDEAL_H = (1 / np.sqrt(2)) * np.array([
    [1, 1],
    [1, -1]
], dtype=complex)

IDEAL_CX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

IDEAL_CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)


def kraus_t1_t2_idle(time: float, t1: float, t2: float) -> list[np.ndarray]:
    """
    Generates Kraus operators for combined T1 and T2 thermal relaxation.
    Note: This assumes T_phi is calculated from T1 and T2.
    """
    if time == 0:
        return [PAULI_I]

    t1_channel = kraus_amplitude_damping(time, t1)
    t2_channel = kraus_dephasing(time, t2, t1)

    # Since T1 and T2 channels (in this formulation) commute,
    # we can combine them by composing their Kraus operators.
    return combine_kraus_channels(t1_channel, t2_channel)


def kraus_zz_interaction(strength: float, time: float) -> list[np.ndarray]:
    """
    Generates a Kraus operator for a ZZ stray interaction.
    This is a unitary evolution U = exp(-i * strength * Z @ Z * time).
    """
    zz = np.kron(PAULI_Z, PAULI_Z)
    # The interaction angle theta = strength * time
    # The scipy.linalg.expm function is suitable for matrix exponentials.
    from scipy.linalg import expm
    unitary = expm(-1j * strength * time * zz)
    return [unitary]


def combine_kraus_channels(
    channel1: list[np.ndarray], channel2: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Combines two quantum channels by composing their Kraus operators.
    This represents applying channel2 after channel1.

    Args:
        channel1: List of Kraus operators for the first channel.
        channel2: List of Kraus operators for the second channel.

    Returns:
        A new list of Kraus operators for the combined channel.
    """
    return [np.dot(k2, k1) for k2 in channel2 for k1 in channel1]


def kraus_heating_to_2(prob: float) -> list[np.ndarray]:
    """
    Generates Kraus operators for a heating channel from |1> to |2>.
    This is a simplified model for leakage heating.

    Args:
        prob: The probability of the |1> -> |2> transition.

    Returns:
        A list of Kraus operators [E0, E1].
    """
    # Operator for the transition |1> -> |2>
    e1 = np.sqrt(prob) * P12.T.conj() # This is |2><1|
    # Operator for remaining in the subspace spanned by |0> and |1>
    e0 = sqrtm(QUTRIT_I - e1.T.conj() @ e1)
    return [e0, e1]


def kraus_cz_leakage(prob: float) -> list[np.ndarray]:
    """
    Generates Kraus operators for dephasing-induced leakage during a CZ gate,
    specifically the |11> -> |02> transition.

    Args:
        prob: The probability of this leakage event.

    Returns:
        A list of Kraus operators for the two-qutrit system.
    """
    # Projector for the |11> state
    p11 = np.kron(P1, P1)
    # Transition operator for |02><11|
    t_02_11 = np.kron(P01.T.conj(), P21.T.conj())

    e1 = np.sqrt(prob) * t_02_11
    e0 = sqrtm(np.eye(9, dtype=complex) - e1.T.conj() @ e1)
    return [e0, e1]
