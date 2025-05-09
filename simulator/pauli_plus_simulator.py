import stim
import numpy as np
from scipy.stats import multivariate_normal
import random

class PauliPlusSimulator:
    def __init__(self, config):
        # Config parameters with defaults
        self.depolarization = config.get("depolarization", 0.001)
        self.leakage_rate = config.get("leakage_rate", 0.01)
        self.cross_talk = config.get("cross_talk", 0.002)
        self.t1 = config.get("t1", 1000)
        self.measurement_duration = config.get("measurement_duration", 100)
        
        # Possible values like Google's experiment
        self.possible_distances = [3, 5]  # Could add 7,9,11 etc.
        self.possible_rounds = list(range(1, 26, 2))  # 1,3,...,25
        self.possible_bases = ['X', 'Z']
    
    def generate_experiment_config(self):
        """Generate a random experiment configuration"""
        return {
            "distance": random.choice(self.possible_distances),
            "rounds": random.choice(self.possible_rounds),
            "basis": random.choice(self.possible_bases)
        }
    
    def generate_experiment(self, config=None):
        """Generate a circuit with random parameters if none provided"""
        if config is None:
            config = self.generate_experiment_config()
            
        self.distance = config["distance"]
        self.rounds = config["rounds"]
        self.basis = config["basis"]
        
        self.circuit = self._build_noisy_circuit(self.basis)
        return self.circuit
    
    def _build_noisy_circuit(self, basis):
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

    def generate_batch(self, num_experiments):
        """Generate multiple experiments with random configurations"""
        return [self.generate_experiment() for _ in range(num_experiments)]