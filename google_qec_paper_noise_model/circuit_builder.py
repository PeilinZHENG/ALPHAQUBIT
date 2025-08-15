# -*- coding: utf-8 -*-
"""
Builds a stim.Circuit for a surface code experiment with a "Pauli+" noise model.

This module implements the "Pauli+" noise model, which includes depolarization
after all Clifford gates, and additional leakage and crosstalk noise after
two-qubit gates. The implementation is aligned with the one found in the
`simulator/pauli_plus_simulator.py` file in this repository.
"""

from pathlib import Path
import yaml
import stim

# The path to the configuration files for this noise model.
CONFIG_DIR = Path(__file__).parent


class SurfaceCodeCircuitBuilder:
    """
    Constructs a noisy stim.Circuit for a surface code experiment using the
    "Pauli+" noise model.
    """

    def __init__(self, distance: int, rounds: int, basis: str = 'Z', processor: str = "pauli_plus"):
        self.distance = distance
        self.rounds = rounds
        self.basis = basis.lower()
        self.processor_name = processor
        self.noise_params = self._load_noise_params()

    def _load_noise_params(self) -> dict:
        """Loads the noise parameters for the specified processor."""
        params_path = CONFIG_DIR / "noise_params.yaml"
        with open(params_path, 'r') as f:
            all_params = yaml.safe_load(f)
        return all_params["processors"][self.processor_name]

    def _attach_noise_to_two_qubit_gates(self, circuit: stim.Circuit) -> stim.Circuit:
        """Insert leakage and cross-talk noise after each two-qubit gate."""
        noisy = stim.Circuit()
        leakage_rate = self.noise_params["leakage_rate"]
        cross_talk = self.noise_params["cross_talk"]

        for inst in circuit:
            noisy.append(inst)
            if inst.name in ("CX", "CZ"):
                targets = [t.value for t in inst.targets_copy()]
                # Leakage modeled as single-qubit Pauli errors on the target qubit
                noisy.append_operation(
                    "PAULI_CHANNEL_2",
                    targets,
                    [
                        leakage_rate / 3,  # IX
                        leakage_rate / 3,  # IY
                        leakage_rate / 3,  # IZ
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                )
                # Crosstalk modeled as a two-qubit depolarizing channel
                noisy.append_operation("DEPOLARIZE2", targets, cross_talk)
        return noisy

    def build_circuit(self) -> stim.Circuit:
        """
        Builds the full, noisy stim.Circuit for the Pauli+ model.
        """
        # 1. Generate a surface code circuit with depolarization after each Clifford gate.
        depolarization = self.noise_params["depolarization"]
        circuit = stim.Circuit.generated(
            f"surface_code:rotated_memory_{self.basis}",
            rounds=self.rounds,
            distance=self.distance,
            after_clifford_depolarization=depolarization,
        )

        # 2. Attach additional leakage and crosstalk noise to two-qubit gates.
        noisy_circuit = self._attach_noise_to_two_qubit_gates(circuit)

        return noisy_circuit


if __name__ == '__main__':
    # Example usage:
    builder = SurfaceCodeCircuitBuilder(distance=3, rounds=10, basis='z')

    # Load parameters
    depolarization = builder.noise_params["depolarization"]
    leakage = builder.noise_params["leakage_rate"]
    crosstalk = builder.noise_params["cross_talk"]
    print(f"Loaded {builder.processor_name} processor params.")
    print(f"Depolarization: {depolarization}, Leakage: {leakage}, Crosstalk: {crosstalk}")

    # Build the noisy circuit
    my_circuit = builder.build_circuit()
    print("\n--- Circuit Stats ---")
    print(my_circuit.stats())
