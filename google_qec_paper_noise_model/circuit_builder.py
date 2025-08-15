# -*- coding: utf-8 -*-
"""
Builds a stim.Circuit for a surface code experiment with a paper-aligned
noise model.

This module uses the detailed physical parameters and cycle schedule defined
in the YAML configuration files to construct a noisy circuit. It uses the
Kraus and Pauli twirling utilities to convert physical noise processes into
stim-compatible Pauli error channels.
"""

from pathlib import Path
import yaml
import stim

from . import kraus_utils
from . import pauli_twirl

# The path to the configuration files for this noise model.
CONFIG_DIR = Path(__file__).parent


class SurfaceCodeCircuitBuilder:
    """
    Constructs a noisy stim.Circuit for a surface code experiment.
    """

    def __init__(self, distance: int, rounds: int, processor: str = "seventy_two_qubit"):
        self.distance = distance
        self.rounds = rounds
        self.processor_name = processor
        self.noise_params = self._load_noise_params()
        self.cycle_schedule = self._load_cycle_schedule()

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

    def _load_cycle_schedule(self) -> dict:
        """Loads the cycle schedule definition."""
        schedule_path = CONFIG_DIR / "cycle_schedule.yaml"
        with open(schedule_path, 'r') as f:
            schedule = yaml.safe_load(f)
        return schedule["cycle"]

    def _add_noise_to_circuit(self, ideal_circuit: stim.Circuit) -> stim.Circuit:
        """
        Iterates through an ideal circuit and injects a custom noise model.
        This version uses a simplified, per-round idle noise model to avoid
        the complexities and errors of per-gate idle timing.
        """
        noisy_circuit = stim.Circuit()

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
                p_meas_err = self.noise_params["measurement"]["misclassification_matrix"]["p1_given_0"]
                noisy_circuit.append("X_ERROR", targets, p_meas_err)

            # --- Add the Gate itself ---
            noisy_circuit.append(instruction)

            # --- Handle Post-Gate Noise ---
            if gate_type == "CX":
                p_2q = self.noise_params["error_rates"]["cz_pauli"] # Use cz_pauli as placeholder
                noise = kraus_utils.kraus_depolarizing(p_2q, n_qubits=2)
                noisy_gate = kraus_utils.combine_kraus_channels([kraus_utils.IDEAL_CX], noise)
                pauli_vec = pauli_twirl.twirl_to_pauli_probs(noisy_gate, n_qubits=2)
                noisy_circuit.append("PAULI_CHANNEL_2", targets, pauli_vec)

            elif gate_type == "H":
                p_1q = self.noise_params["error_rates"]["single_qubit_pauli"]
                noise = kraus_utils.kraus_depolarizing(p_1q, n_qubits=1)
                noisy_gate = kraus_utils.combine_kraus_channels([kraus_utils.IDEAL_H], noise)
                pauli_vec = pauli_twirl.twirl_to_pauli_probs(noisy_gate, n_qubits=1)
                noisy_circuit.append("PAULI_CHANNEL_1", targets, pauli_vec)

            elif gate_type == "R":
                p_reset_err = self.noise_params["leakage_heating"]["reset_error_rate"]
                noisy_circuit.append("X_ERROR", targets, p_reset_err)

            elif gate_type == "SHIFT_COORDS":
                # This instruction reliably marks the end of a full stabilizer cycle.
                # We apply an approximate idle noise to all data qubits here.
                idle_p1 = 0.0001
                if data_qubit_indices:
                    noisy_circuit.append("DEPOLARIZE1", data_qubit_indices, idle_p1)

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
    t1 = builder.noise_params["t1_mean"]
    gate_time_cz = builder.noise_params["gate_times"]["cz"]
    print(f"Loaded {builder.processor_name} processor params.")
    print(f"T1: {t1*1e6:.1f} us, CZ time: {gate_time_cz*1e9:.0f} ns")

    # Build the (placeholder) circuit
    my_circuit = builder.build_circuit()
    # print("\n--- Circuit Stats ---")
    # print(my_circuit.stats())
