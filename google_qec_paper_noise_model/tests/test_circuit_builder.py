# -*- coding: utf-8 -*-
"""
Integration tests for the circuit_builder module.
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
    Tests that the generated circuit contains the expected types of instructions,
    including the injected noise channels.
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

    # Check for our custom injected noise
    assert "PAULI_CHANNEL_1" in circuit_str
    assert "PAULI_CHANNEL_2" in circuit_str

    # Check for measurement and reset noise
    # We model this with X_ERROR for now
    assert "X_ERROR" in circuit_str

def test_circuit_stats_are_reasonable():
    """
    Checks the stats of a small generated circuit to ensure the number of
    operations is reasonable with the refactored noise model.
    """
    distance = 3
    rounds = 2
    builder = circuit_builder.SurfaceCodeCircuitBuilder(distance=distance, rounds=rounds)
    circuit = builder.build_circuit()
    circuit_str = str(circuit)

    ideal_circuit = circuit_builder.stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
    )

    def count_ops(circuit, name):
        return sum(1 for op in circuit if op.name == name)

    # The number of CXs and Hs should be the same as the ideal circuit.
    assert count_ops(circuit, "CX") == count_ops(ideal_circuit, "CX")
    assert count_ops(circuit, "H") == count_ops(ideal_circuit, "H")

    # The number of PAULI_CHANNEL_2 instructions should match the number of CXs.
    assert circuit_str.count("PAULI_CHANNEL_2") == count_ops(ideal_circuit, "CX")

    # The number of PAULI_CHANNEL_1 instructions should match the number of Hs.
    assert circuit_str.count("PAULI_CHANNEL_1") == count_ops(ideal_circuit, "H")

    # The number of DEPOLARIZE1 instructions should match the number of rounds - 1.
    # (SHIFT_COORDS appears after round 1, not round 0).
    assert circuit_str.count("DEPOLARIZE1") == rounds - 1
