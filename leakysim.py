"""Stub module used for tests when leakysim is unavailable."""

class ChannelStub:
    def get_prob_from_to(self, status_from, status_to, pauli):
        return 0.0

def generalized_pauli_twirling(kraus_ops, num_qubits, num_level, safety_check=False):
    return ChannelStub()

class LeakageStatus:
    def __init__(self, status):
        self.status = status
