import numpy as np
import stim

from simulator.si1000_generator import si1000_noise_model
from simulator.pauli_plus_simulator import PauliPlusSimulator


def test_si1000_zero_noise():
    config = {"p": 0.0, "distance": 3, "rounds": 3}
    circuit = si1000_noise_model(config)
    sampler = circuit.compile_detector_sampler()
    syndromes, logicals = sampler.sample(100, separate_observables=True)
    assert np.all(syndromes == 0)
    assert np.all(logicals == 0)


def test_pauli_plus_noise_attached_to_gates():
    config = {
        "distance": 3,
        "rounds": 3,
        "depolarization": 0.0,
        "leakage_rate": 0.0,
        "cross_talk": 0.0,
    }
    sim = PauliPlusSimulator(config, 'z')
    ops = list(sim.circuit)
    for idx, inst in enumerate(ops):
        if inst.name in ("CX", "CZ"):
            next1 = ops[idx + 1]
            next2 = ops[idx + 2]
            assert next1.name == "PAULI_CHANNEL_2"
            assert next2.name == "DEPOLARIZE2"
            break
    else:
        assert False, "No two-qubit gate found in circuit"

    sampler = sim.circuit.compile_detector_sampler()
    syndromes, logicals = sampler.sample(100, separate_observables=True)
    assert np.all(syndromes == 0)
    assert np.all(logicals == 0)
