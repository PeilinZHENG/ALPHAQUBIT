import stim


def si1000_noise_model(config: dict) -> stim.Circuit:
    """Construct a surface-code circuit using Stim's SI1000 noise model."""
    p = config.get("p", 0.001)
    distance = config.get("distance", 3)
    rounds = config.get("rounds", 25)
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p / 10,
        before_round_data_depolarization=p / 10,
        before_measure_flip_probability=5 * p,
        after_reset_flip_probability=5 * p,
    )
    return circuit
