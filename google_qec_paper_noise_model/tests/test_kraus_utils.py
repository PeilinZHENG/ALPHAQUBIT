# -*- coding: utf-8 -*-
"""
Unit tests for the kraus_utils module.
"""

import numpy as np
import pytest

from .. import kraus_utils


def check_kraus_is_trace_preserving(kraus_ops: list[np.ndarray]):
    """
    Helper function to check if a set of Kraus operators for a
    single-qubit channel is trace-preserving (sum K_dag * K = I).
    """
    d = kraus_ops[0].shape[0]
    identity = np.identity(d, dtype=complex)
    sum_k_dag_k = np.zeros((d, d), dtype=complex)
    for k in kraus_ops:
        sum_k_dag_k += k.conj().T @ k

    np.testing.assert_allclose(sum_k_dag_k, identity, atol=1e-9)


def test_amplitude_damping_is_trace_preserving():
    """
    Tests that the amplitude damping channel is trace-preserving.
    """
    kraus_ops = kraus_utils.kraus_amplitude_damping(time=50e-9, t1=20e-6)
    check_kraus_is_trace_preserving(kraus_ops)


def test_dephasing_is_trace_preserving():
    """
    Tests that the dephasing channel is trace-preserving.
    """
    kraus_ops = kraus_utils.kraus_dephasing(time=50e-9, t2=30e-6, t1=20e-6)
    check_kraus_is_trace_preserving(kraus_ops)


def test_depolarizing_1q_is_trace_preserving():
    """
    Tests that the 1Q depolarizing channel is trace-preserving.
    """
    kraus_ops = kraus_utils.kraus_depolarizing(prob=0.01, n_qubits=1)
    check_kraus_is_trace_preserving(kraus_ops)
