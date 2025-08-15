# -*- coding: utf-8 -*-
"""
Integration tests for the circuit_builder module, specifically for the
"Pauli+" noise model.
"""

import pytest
from .. import circuit_builder


def test_circuit_builder_builds_a_circuit():
    """
    Tests that the builder creates a stim.Circuit object without errors.
    """
    builder = circuit_builder.SurfaceCodeCircuitBuilder(distance=3, rounds=2)
    circuit = builder.build_circuit()

    assert circuit is not None
    assert isinstance(circuit, circuit_builder.stim.Circuit)


def test_circuit_contains_expected_instructions():
    """
    Tests that the generated circuit contains the expected types of instructions
    for the Pauli+ noise model.
    """
    builder = circuit_builder.SurfaceCodeCircuitBuilder(distance=3, rounds=2)
    circuit = builder.build_circuit()

    circuit_str = str(circuit)

    # Check for core surface code instructions
    assert "H" in circuit_str
    assert "CX" in circuit_str
    assert "M" in circuit_str
    assert "R" in circuit_str
    assert "DETECTOR" in circuit_str
    assert "OBSERVABLE_INCLUDE" in circuit_str

    # Check for our injected noise channels
    assert "DEPOLARIZE1" in circuit_str  # From after_clifford_depolarization
    assert "DEPOLARIZE2" in circuit_str  # From crosstalk
    assert "PAULI_CHANNEL_2" in circuit_str  # From leakage


def test_circuit_stats_are_reasonable():
    """
    Checks the stats of a small generated circuit to ensure the number of
    noise operations is reasonable with the Pauli+ model.
    """
    distance = 3
    rounds = 2
    builder = circuit_builder.SurfaceCodeCircuitBuilder(distance=distance, rounds=rounds)
    circuit = builder.build_circuit()

    def count_ops(c, name):
        return sum(1 for op in c if op.name == name)

    # Count two-qubit gates in the noisy circuit
    num_two_qubit_gates = count_ops(circuit, "CX") + count_ops(circuit, "CZ")

    # Check that leakage channels are added for each 2Q gate
    assert count_ops(circuit, "PAULI_CHANNEL_2") == num_two_qubit_gates

    # Check that crosstalk and depolarization channels are added for each 2Q gate.
    # The `after_clifford_depolarization` adds one DEPOLARIZE2, and the
    # crosstalk implementation adds another one.
    assert count_ops(circuit, "DEPOLARIZE2") == 2 * num_two_qubit_gates

    # Check that DEPOLARIZE1 is present for single-qubit gates
    # The exact number is complex to calculate, so we check for presence
    assert count_ops(circuit, "DEPOLARIZE1") > 0
