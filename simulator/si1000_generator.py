import stim

def si1000_noise_model(p: float) -> stim.Circuit:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=25,
        distance=3,
        after_clifford_depolarization=p/10,
        after_reset_flip_probability=5*p,
    )
    return circuit