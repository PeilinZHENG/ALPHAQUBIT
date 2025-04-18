import stim

def generate_dem_data(num_samples: int, dem_config: dict):
    circuit = stim.Circuit.generated(
        dem_config["circuit_params"]["code_task"],  # Single string argument
        distance=dem_config["circuit_params"]["distance"],
        rounds=dem_config["circuit_params"]["rounds"],
        after_clifford_depolarization=dem_config["error_rates"]["after_clifford_depolarization"],
        after_reset_flip_probability=dem_config["error_rates"]["after_reset_flip_probability"],
    )
    sampler = circuit.compile_detector_sampler()
    syndromes, logical_errors = sampler.sample(num_samples, separate_observables=True)
    return syndromes, logical_errors