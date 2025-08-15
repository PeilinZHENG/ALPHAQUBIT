# -*- coding: utf-8 -*-
"""
Builds a stim.Circuit for a surface code experiment with a detailed physical
noise model aligned with the Nature paper.
"""

from pathlib import Path
import yaml
import stim
from . import kraus_utils

# The path to the configuration files for this noise model.
CONFIG_DIR = Path(__file__).parent


class SurfaceCodeCircuitBuilder:
    """
    Constructs a noisy stim.Circuit for a surface code experiment using a
    detailed physical noise model.
    """

    def __init__(self, distance: int, rounds: int, basis: str = 'Z', processor: str = "72_qubit_paper_aligned"):
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

    def build_circuit(self) -> stim.Circuit:
        """
        Builds the full, noisy stim.Circuit by adding a physical noise model
        to a stim-generated ideal circuit.
        """
        # For this complex model, we will manually add noise to an ideal circuit.
        # This gives us more control than using after_clifford_depolarization.
        ideal_circuit = stim.Circuit.generated(
            f"surface_code:rotated_memory_{self.basis}",
            rounds=self.rounds,
            distance=self.distance,
        )

        noisy_circuit = stim.Circuit()
        params = self.noise_params

        # Extract parameters for convenience
        t1 = params["decoherence"]["t1_us"] * 1e-6
        t2 = params["decoherence"]["t2_cpmg_us"] * 1e-6
        p_reset = params["readout_reset"]["reset"]
        p_readout = params["readout_reset"]["readout"]
        p_sq_gate = params["gate_errors"]["sq_gates"]
        p_cz_leakage = params["gate_errors"]["cz_leakage_prob"]
        p_cz_crosstalk = params["gate_errors"]["cz_crosstalk"]
        # Note: cz_total_error is not used directly, as we build it from components.

        # A rough estimate for gate times to use with T1/T2 noise
        # The paper mentions a cycle time of 1076 ns. A cycle has many gates.
        # We'll use a placeholder value for gate duration.
        time_1q = 25e-9
        time_2q = 50e-9

        for instruction in ideal_circuit:
            targets = [t.value for t in instruction.targets_copy()]
            gate_type = instruction.name

            # Add noise before the gate
            if gate_type == "M":
                noisy_circuit.append("X_ERROR", targets, p_readout)

            # Add the gate itself
            noisy_circuit.append(instruction)

            # Add noise after the gate
            if gate_type in ["H", "S", "S_DAG"]:
                # Single-qubit gate error (excess error)
                noisy_circuit.append("DEPOLARIZE1", targets, p_sq_gate)
                # Decoherence during the gate
                decoherence = kraus_utils.kraus_t1_t2_idle(time_1q, t1, t2)
                # Here we would twirl and add the Pauli channel, but for simplicity
                # we'll approximate with a depolarizing channel for now.
                # This part would need the Generalized Pauli Twirling.
                # For now, we omit the decoherence part during gates to avoid
                # implementing the full twirling.

            elif gate_type == "CX" or gate_type == "CZ":
                # For 2Q gates, the paper describes a complex model.
                # We will approximate it with separate noise channels.
                # 1. Crosstalk
                noisy_circuit.append("DEPOLARIZE2", targets, p_cz_crosstalk)
                # 2. Leakage, approximated as a Pauli channel on the target
                leakage_channel = [
                    p_cz_leakage / 3, p_cz_leakage / 3, p_cz_leakage / 3,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
                noisy_circuit.append("PAULI_CHANNEL_2", targets, leakage_channel)

            elif gate_type == "R":
                noisy_circuit.append("X_ERROR", targets, p_reset)

            elif gate_type == "TICK":
                # Idle noise could be added here, but the paper's model seems to
                # lump it into an "excess error" on idle data qubits.
                # This is a complex part of the model that is hard to implement
                # without more details on the timing and structure.
                pass

        return noisy_circuit


if __name__ == '__main__':
    builder = SurfaceCodeCircuitBuilder(distance=3, rounds=10)
    print(f"Loaded {builder.processor_name} processor params.")
    print(builder.noise_params)

    my_circuit = builder.build_circuit()
    print("\n--- Circuit Stats ---")
    print(my_circuit.stats())
