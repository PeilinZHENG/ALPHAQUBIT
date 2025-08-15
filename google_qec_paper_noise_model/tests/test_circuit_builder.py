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

    # Check for our injected noise channels
    assert "DEPOLARIZE1" in circuit_str
    assert "DEPOLARIZE2" in circuit_str
    assert "X_ERROR" in circuit_str


def test_circuit_stats_are_reasonable():
    """
    Checks the stats of a small generated circuit to ensure the number of
    operations is reasonable with the new noise model.
    """
    distance = 3
    rounds = 2
    builder = circuit_builder.SurfaceCodeCircuitBuilder(distance=distance, rounds=rounds)
    circuit = builder.build_circuit()

    ideal_circuit = circuit_builder.stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
    )

    def count_ops(c, name):
        return sum(1 for op in c if op.name == name)

    # The number of core gate operations should be the same as the ideal circuit.
    assert count_ops(circuit, "CX") == count_ops(ideal_circuit, "CX")
    assert count_ops(circuit, "H") == count_ops(ideal_circuit, "H")
    assert count_ops(circuit, "M") == count_ops(ideal_circuit, "M")
    assert count_ops(circuit, "R") == count_ops(ideal_circuit, "R")

    # Check gate noise channels
    num_single_qubit_gates = sum(
        1 for op in ideal_circuit if op.name in ["H", "S", "S_DAG"]
    )
    num_two_qubit_gates = count_ops(ideal_circuit, "CX")

    # Count DEPOLARIZE1 and DEPOLARIZE2 instructions associated with gates
    gate_depolarize1_count = 0
    gate_depolarize2_count = 0
    for i, op in enumerate(circuit):
        if i > 0:
            prev_op = circuit[i-1]
            if prev_op.name in ["H", "S", "S_DAG"] and op.name == "DEPOLARIZE1":
                gate_depolarize1_count += 1
            elif prev_op.name == "CX" and op.name == "DEPOLARIZE2":
                gate_depolarize2_count += 1

    assert gate_depolarize1_count == num_single_qubit_gates
    assert gate_depolarize2_count == num_two_qubit_gates

    # Check reset and measurement noise
    assert count_ops(circuit, "X_ERROR") == (
        count_ops(ideal_circuit, "R") + count_ops(ideal_circuit, "M")
    )

    # Check idle noise (applied once per round, but not after the last round)
    total_depolarize1 = count_ops(circuit, "DEPOLARIZE1")
    idle_depolarize_count = total_depolarize1 - gate_depolarize1_count
    assert idle_depolarize_count == max(0, rounds - 1)
