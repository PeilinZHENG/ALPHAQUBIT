import random
from pathlib import Path
from typing import Optional, Union
import numpy as np
import stim

def create_sampler(stim_path: Path, dem_path: Optional[Path]):
    """Create a Stim sampler from either a DEM file or Circuit."""
    if dem_path and dem_path.exists():
        dem = stim.DetectorErrorModel.from_file(dem_path)
    else:
        circuit = stim.Circuit.from_file(stim_path)
        try:
            # Try new parameter name first
            dem = circuit.detector_error_model(ignore_decomposition_failures=True)
        except TypeError:
            # Fall back to old parameter name
            dem = circuit.detector_error_model(ignore_decomposition_errors=True)

    try:
        return dem.compile_sampler(allow_gauge_detectors=True)
    except TypeError:
        # Fallback for older Stim versions
        return dem.compile_sampler()

def sample_circuit(stim_path: Path, dem_path: Optional[Path], shots: int) -> np.ndarray:
    """Sample a quantum circuit and return as uint8 numpy array."""
    sampler = create_sampler(stim_path, dem_path)
    
    # Try different sampling methods with proper error handling
    try:
        # Method 1: Newest API with random generator
        rng = np.random.Generator(np.random.PCG64(random.randrange(2**32)))
        result = sampler.sample(shots=shots, random_generator=rng)
    except TypeError:
        try:
            # Method 2: Older API with seed parameter
            result = sampler.sample(shots, seed=random.randrange(2**32))
        except TypeError:
            # Method 3: Simplest API
            result = sampler.sample(shots)
    
    # Convert result to numpy array if it isn't already
    if isinstance(result, tuple):
        # Handle case where sample() returns (detectors, observables)
        detectors = np.array(result[0], dtype=np.uint8)
        return detectors
    elif hasattr(result, 'astype'):
        return result.astype(np.uint8)
    else:
        return np.array(result, dtype=np.uint8)

def test_sampler():
    """Test the sampling functionality."""
    test_stim = Path("C:\\Users\\Lenovo\\Downloads\\google_qec3v5_experiment_data\\surface_code_bZ_d3_r25_center_5_3\\circuit_ideal.stim")
    if not test_stim.exists():
        # Create a minimal test circuit if none exists
        print("Creating test circuit...")
        circuit = stim.Circuit()
        circuit.append_operation("H", [0])
        circuit.append_operation("CNOT", [0, 1])
        circuit.append_operation("M", [0, 1])
        circuit.to_file(test_stim)
        print(f"Created test circuit at {test_stim}")
    
    try:
        print("Testing sampler with 10 shots...")
        samples = sample_circuit(test_stim, None, shots=10)
        print("Sampling successful!")
        print(f"Sample output shape: {samples.shape}")
        print("First 5 samples:")
        print(samples[:5] if len(samples) > 5 else samples)
    except Exception as e:
        print(f"Sampling failed: {type(e).__name__} - {str(e)}")
        raise

if __name__ == "__main__":
    test_sampler()
 