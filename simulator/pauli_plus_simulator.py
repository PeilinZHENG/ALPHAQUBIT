import stim
import numpy as np
from scipy.stats import multivariate_normal

class PauliPlusSimulator:
    def __init__(self, config, basis):
        self.distance = config["distance"]
        self.rounds = config["rounds"]
        self.depolarization = config["depolarization"]
        self.leakage_rate = config["leakage_rate"]
        self.cross_talk = config["cross_talk"]
        self.t1 = config["t1"]
        self.measurement_duration = config["measurement_duration"]
        self.circuit = self._build_noisy_circuit(basis)

    def _build_noisy_circuit(self,basis):
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_"+basis,
            rounds=self.rounds,
            distance=self.distance,
            after_clifford_depolarization=self.depolarization
        )
        circuit = self._add_leakage_to_cz(circuit)
        circuit = self._add_cross_talk(circuit)
        return circuit

    def _add_leakage_to_cz(self, circuit):
        """Add leakage noise to CZ gates with proper PAULI_CHANNEL_2 parameters"""
        for i in range(self.distance**2 - 1):
            circuit.append_operation("PAULI_CHANNEL_2", [i, i+1], [
                # 15 parameters for all non-identity Pauli combinations
                self.leakage_rate/3,  # IX
                self.leakage_rate/3,  # IY
                self.leakage_rate/3,  # IZ
                0, 0, 0, 0,           # XI, XX, XY, XZ
                0, 0, 0, 0,           # YI, YX, YY, YZ
                0, 0, 0, 0            # ZI, ZX, ZY, ZZ
            ])
        return circuit

    def _add_cross_talk(self, circuit):
        for q in range(self.distance**2 - 1):
            neighbors = self._get_adjacent_qubits(q)
            for n in neighbors:
                if n > q:
                    circuit.append_operation("DEPOLARIZE2", [q, n], self.cross_talk)
        return circuit

    def _get_adjacent_qubits(self, q):
        row = q // self.distance
        col = q % self.distance
        neighbors = []
        if col > 0: neighbors.append(q - 1)
        if col < self.distance - 1: neighbors.append(q + 1)
        if row > 0: neighbors.append(q - self.distance)
        if row < self.distance - 1: neighbors.append(q + self.distance)
        return neighbors