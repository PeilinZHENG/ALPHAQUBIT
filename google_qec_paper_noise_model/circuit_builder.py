# -*- coding: utf-8 -*-
"""
Builds a stim.Circuit for a surface code experiment with a circuit-level
depolarizing noise model.

This module uses the noise parameters defined in the YAML configuration
files to construct a noisy circuit based on a standard depolarizing model.
"""

from pathlib import Path
import yaml
import stim

# The path to the configuration files for this noise model.
CONFIG_DIR = Path(__file__).parent


class SurfaceCodeCircuitBuilder:
    """
    Constructs a noisy stim.Circuit for a surface code experiment.
    """

    def __init__(self, distance: int, rounds: int, processor: str = "uniform_depolarizing"):
        self.distance = distance
        self.rounds = rounds
        self.processor_name = processor
        self.noise_params = self._load_noise_params()

    def _get_data_qubit_coords(self) -> list[complex]:
        """Returns a list of complex coordinates for the data qubits."""
        data_qubits = []
        for r in range(self.distance):
            for c in range(self.distance):
                data_qubits.append(complex(2 * r + 1, 2 * c + 1))
        return data_qubits

    def _load_noise_params(self) -> dict:
        """Loads the noise parameters for the specified processor."""
        params_path = CONFIG_DIR / "noise_params.yaml"
        with open(params_path, 'r') as f:
            all_params = yaml.safe_load(f)
        return all_params["processors"][self.processor_name]

    def _add_noise_to_circuit(self, ideal_circuit: stim.Circuit) -> stim.Circuit:
        """
        Iterates through an ideal circuit and injects a circuit-level
        depolarizing noise model.
        """
        noisy_circuit = stim.Circuit()
        p_gate = self.noise_params["p_gate"]
        p_meas = self.noise_params["p_meas"]
        p_reset = self.noise_params["p_reset"]
        p_idle = self.noise_params["p_idle"]

        # Build a map from complex coordinates to integer indices from the ideal circuit
        coord_to_index = {}
        for instruction in ideal_circuit:
            if instruction.name == "QUBIT_COORDS":
                coords = instruction.gate_args_copy()
                qubit_index = instruction.targets_copy()[0].value
                coord_to_index[complex(coords[0], coords[1])] = qubit_index

        # Get the integer indices for all data qubits
        data_qubit_coords = self._get_data_qubit_coords()
        data_qubit_indices = [
            coord_to_index[c] for c in data_qubit_coords if c in coord_to_index
        ]

        for instruction in ideal_circuit:
            targets = [t.value for t in instruction.targets_copy()]
            gate_type = instruction.name

            # --- Handle Pre-Gate Noise (e.g., Measurement) ---
            if gate_type == "M":
                noisy_circuit.append("X_ERROR", targets, p_meas)

            # --- Add the Gate itself ---
            noisy_circuit.append(instruction)

            # --- Handle Post-Gate Noise ---
            if gate_type in ["H", "S", "S_DAG"]:
                noisy_circuit.append("DEPOLARIZE1", targets, p_gate)
            elif gate_type == "CX":
                noisy_circuit.append("DEPOLARIZE2", targets, p_gate)
            elif gate_type == "R":
                noisy_circuit.append("X_ERROR", targets, p_reset)
            elif gate_type == "SHIFT_COORDS":
                # This instruction reliably marks the end of a full stabilizer cycle.
                # We apply idle noise to all data qubits here.
                if data_qubit_indices:
                    noisy_circuit.append("DEPOLARIZE1", data_qubit_indices, p_idle)

        return noisy_circuit

    def build_circuit(self) -> stim.Circuit:
        """
        Builds the full, noisy stim.Circuit by adding noise to a
        stim-generated ideal circuit.
        """
        # 1. Generate a perfect, noiseless surface code circuit using stim.
        ideal_circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=self.rounds,
            distance=self.distance,
        )

        # 2. Add our custom noise model to the ideal circuit.
        noisy_circuit = self._add_noise_to_circuit(ideal_circuit)

        return noisy_circuit


if __name__ == '__main__':
    # Example usage:
    builder = SurfaceCodeCircuitBuilder(distance=3, rounds=10)

    # Load parameters
    p_gate = builder.noise_params["p_gate"]
    p_idle = builder.noise_params["p_idle"]
    print(f"Loaded {builder.processor_name} processor params.")
    print(f"p_gate: {p_gate}, p_idle: {p_idle}")

    # Build the noisy circuit
    my_circuit = builder.build_circuit()
    print("\n--- Circuit Stats ---")
    print(my_circuit.stats())
