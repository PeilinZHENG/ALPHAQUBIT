import stim


class PauliPlusSimulator:
    """Simulator adding leakage and cross-talk noise after two-qubit gates."""

    def __init__(self, config, basis: str):
        self.distance = config.get("distance")
        self.rounds = config.get("rounds")
        if self.distance is None or self.rounds is None:
            raise ValueError("config must include 'distance' and 'rounds'")

        self.depolarization = config.get("depolarization", 0.001)
        self.leakage_rate = config.get("leakage_rate", 0.01)
        self.cross_talk = config.get("cross_talk", 0.002)

        self.basis = basis.upper()
        if self.basis not in ("X", "Z"):
            raise ValueError("basis must be 'X' or 'Z'")

        self.circuit = self._build_noisy_circuit()

    def _build_noisy_circuit(self) -> stim.Circuit:
        circuit = stim.Circuit.generated(
            f"surface_code:rotated_memory_{self.basis.lower()}",
            rounds=self.rounds,
            distance=self.distance,
            after_clifford_depolarization=self.depolarization,
        )
        return self._attach_noise_to_two_qubit_gates(circuit)

    def _attach_noise_to_two_qubit_gates(self, circuit: stim.Circuit) -> stim.Circuit:
        """Insert leakage and cross-talk noise after each two-qubit gate."""
        noisy = stim.Circuit()
        for inst in circuit:
            noisy.append(inst)
            if inst.name in ("CX", "CZ"):
                targets = [t.value for t in inst.targets_copy()]
                noisy.append_operation(
                    "PAULI_CHANNEL_2",
                    targets,
                    [
                        self.leakage_rate / 3,
                        self.leakage_rate / 3,
                        self.leakage_rate / 3,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                )
                noisy.append_operation("DEPOLARIZE2", targets, self.cross_talk)
        return noisy

    def apply_paper_aligned_noise(self, config):
        """Adjust noise parameters based on a paper-aligned configuration.

        The paper-aligned model specifies detailed physical error rates. For the
        purposes of this simulator we only map a small subset of those
        parameters onto the existing depolarization, leakage, and cross-talk
        knobs. Any parameters not present in ``config`` retain their existing
        values.
        """
        self.depolarization = config.get("p_1q_excess", self.depolarization)
        self.leakage_rate = config.get("p_cz_leak_11_to_02", self.leakage_rate)
        self.cross_talk = config.get("p_cz_crosstalk_ZZ", self.cross_talk)
        # Rebuild the circuit so that updated parameters take effect.
        self.circuit = self._build_noisy_circuit()
