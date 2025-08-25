# -*- coding: utf-8 -*-
"""
Builds a stim.Circuit for a surface code experiment with a detailed physical
noise model aligned with the Nature paper, using leakysim for Generalized
Pauli Twirling.
"""

from pathlib import Path
import yaml
import stim
import numpy as np
from . import kraus_utils, pauli_twirl
try:
    import leakysim
except ModuleNotFoundError:  # pragma: no cover - fallback for newer package name
    import leaky as leakysim

# The path to the configuration files for this noise model.
CONFIG_DIR = Path(__file__).parent


class SurfaceCodeCircuitBuilder:
    """
    Constructs a noisy stim.Circuit for a surface code experiment using a
    detailed physical noise model and generalized Pauli twirling.
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
        p_sq_gate_err = params["gate_errors"]["sq_gates"]
        p_cz_leakage = params["gate_errors"]["cz_leakage_prob"]
        p_cz_crosstalk = params["gate_errors"]["cz_crosstalk"]

        time_1q = 25e-9
        time_2q = 50e-9

        for instruction in ideal_circuit:
            targets = [t.value for t in instruction.targets_copy()]
            gate_type = instruction.name

            noisy_circuit.append(instruction)

            # --- Construct and add noise channel for this gate ---
            if gate_type in ["H", "S", "S_DAG"]:
                depol = kraus_utils.kraus_depolarizing(p_sq_gate_err, 1)
                deco = kraus_utils.kraus_t1_t2_idle(time_1q, t1, t2)
                channel = kraus_utils.combine_kraus_channels(deco, depol)
                probs = pauli_twirl.twirl_to_pauli_probs(channel, 1)
                noisy_circuit.append("PAULI_CHANNEL_1", targets, probs)

            elif gate_type in ["CX", "CZ"]:
                depol = kraus_utils.kraus_depolarizing(p_cz_crosstalk, 2)
                deco1 = kraus_utils.kraus_t1_t2_idle(time_2q, t1, t2)
                deco2 = kraus_utils.kraus_t1_t2_idle(time_2q, t1, t2)
                deco = [np.kron(k1, k2) for k1 in deco1 for k2 in deco2]
                channel = kraus_utils.combine_kraus_channels(deco, depol)
                probs = pauli_twirl.twirl_to_pauli_probs(channel, 2)
                noisy_circuit.append("PAULI_CHANNEL_2", targets, probs)
                leakage_probs = [0, p_cz_leakage / 3, p_cz_leakage / 3, p_cz_leakage / 3] + [0] * 12
                noisy_circuit.append("PAULI_CHANNEL_2", targets, leakage_probs[1:])

            elif gate_type == "M":
                noisy_circuit.append("X_ERROR", targets, p_readout)

            elif gate_type == "R":
                noisy_circuit.append("X_ERROR", targets, p_reset)

        return noisy_circuit


if __name__ == '__main__':
    builder = SurfaceCodeCircuitBuilder(distance=3, rounds=2)
    print(f"Loaded {builder.processor_name} processor params.")

    my_circuit = builder.build_circuit()
    print("\n--- Circuit Stats ---")
    print(my_circuit.stats())
