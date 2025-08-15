# -*- coding: utf-8 -*-
"""
Unit tests for the pauli_twirl module.
"""

import numpy as np
import pytest

from .. import kraus_utils
from .. import pauli_twirl


def test_twirl_1q_depolarizing_channel():
    """
    Tests Pauli twirling on a 1-qubit depolarizing channel.
    """
    error_prob = 0.03
    n_qubits = 1

    kraus_ops = kraus_utils.kraus_depolarizing(prob=error_prob, n_qubits=n_qubits)
    pauli_error_vector = pauli_twirl.twirl_to_pauli_probs(
        kraus_ops, n_qubits=n_qubits, method="ptm"
    )

    expected_prob_per_pauli = error_prob / 3.0
    expected_vector = np.full(3, expected_prob_per_pauli)

    assert pauli_error_vector.shape == (3,)
    np.testing.assert_allclose(np.sum(pauli_error_vector), error_prob, atol=1e-9)
    np.testing.assert_allclose(pauli_error_vector, expected_vector, atol=1e-9)


def test_twirl_2q_depolarizing_channel():
    """
    Tests Pauli twirling on a 2-qubit depolarizing channel.

    For this channel, the 15 non-identity Pauli errors should be
    equiprobable.
    """
    # Total error probability for the channel
    error_prob = 0.03
    n_qubits = 2

    # 1. Create the Kraus operators for a 2-qubit depolarizing channel.
    kraus_ops = kraus_utils.kraus_depolarizing(prob=error_prob, n_qubits=n_qubits)

    # 2. Use the PTM twirling function to get the Pauli error probabilities.
    pauli_error_vector = pauli_twirl.twirl_to_pauli_probs(
        kraus_ops, n_qubits=n_qubits, method="ptm"
    )

    # 3. Define the expected outcome.
    # The total error `p` should be distributed equally among the 15
    # non-identity Pauli operators.
    expected_prob_per_pauli = error_prob / 15.0
    expected_vector = np.full(15, expected_prob_per_pauli)

    # 4. Assert correctness.
    # Check that the vector has the correct shape.
    assert pauli_error_vector.shape == (15,)

    # Check that the sum of probabilities is correct.
    np.testing.assert_allclose(
        np.sum(pauli_error_vector), error_prob, atol=1e-9
    )

    # Check that the calculated vector matches the expected one.
    np.testing.assert_allclose(
        pauli_error_vector, expected_vector, atol=1e-9
    )

def test_twirl_completeness_check():
    """
    Tests that the sum of Kraus operators K_i^† K_i is the identity.
    """
    error_prob = 0.03
    kraus_ops = kraus_utils.kraus_depolarizing(prob=error_prob, n_qubits=2)

    sum_k_dag_k = np.zeros((4, 4), dtype=complex)
    for k in kraus_ops:
        sum_k_dag_k += k.conj().T @ k

    np.testing.assert_allclose(sum_k_dag_k, np.identity(4), atol=1e-9)

def test_pauli_basis_is_orthonormal():
    """
    Verifies that the Pauli basis used in the twirling code is orthonormal
    under the trace inner product, Tr(A† B) / d.
    """
    d = 4 # Two qubits
    basis = pauli_twirl.PAULI_2Q_BASIS
    for i, p_i in enumerate(basis):
        for j, p_j in enumerate(basis):
            # Inner product is Tr(Pi^† Pj) / d
            inner_product = np.trace(p_i.conj().T @ p_j) / d
            if i == j:
                # Should be 1 on the diagonal
                assert abs(inner_product - 1.0) < 1e-9, f"P_{i} is not normalized"
            else:
                # Should be 0 off the diagonal
                assert abs(inner_product) < 1e-9, f"P_{i} and P_{j} are not orthogonal"
