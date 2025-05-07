from pathlib import Path
from typing import Optional
import numpy as np
import random
import stim
import shutil
from circuit_utils import CircuitUtils
from data_manager import DataManager

class ExperimentSimulator:
    def __init__(self, root_dir: Path, shots: int):
        self.root = root_dir.resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.root}")
        
        self.shots = int(shots)
        if self.shots <= 0:
            raise ValueError("Shots must be positive")
            
        self.data_manager = DataManager()
        self.circuit_utils = CircuitUtils()

    def _sample_circuit(self, stim_path: Path, dem_path: Optional[Path]) -> np.ndarray:
        """Sample a single quantum circuit."""
        try:
            sampler = self.circuit_utils.create_sampler(stim_path, dem_path)
            result = sampler.sample(shots=self.shots)  # Ensure 'shots' is being passed here
            if isinstance(result, tuple):
                return result[0]  # Return just the detectors
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to sample circuit {stim_path}: {str(e)}") from e


    def run_simulation(self) -> None:
        """Run simulations for all circuits found in subdirectories."""
        dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not dirs:
            raise ValueError(f"No circuit directories found in {self.root}")
        
        success_count = 0
        for cid, folder in enumerate(dirs):
            try:
                stim_path, dem_path = self.circuit_utils.locate_stim_and_dem(folder)
                print(f"→ {folder.name}: {self.shots:,} shots (cid={cid})")
                
                samples = self._sample_circuit(stim_path, dem_path)
                self.data_manager.store_results(cid, stim_path, samples)
                success_count += 1
                
            except Exception as e:
                print(f"  ✖ Failed {folder.name}: {type(e).__name__} - {str(e)}")
                continue

        if success_count == 0:
            raise RuntimeError("All simulations failed - check circuit definitions")

    def save_results(self, npz_path: Path, table_path: Path) -> None:
        """Save results to files."""
        self.data_manager.save_results(npz_path, table_path)

def test_experiment_simulator():
    """Test function for ExperimentSimulator"""
    print("\n=== Testing ExperimentSimulator ===")
    test_dir = Path("test_experiment_sim")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Create valid test circuit (repetition code)
        valid_circuit = stim.Circuit("""
            # Initialize qubits
            R 0 1 2
            # Create GHZ state
            H 0
            CNOT 0 1
            CNOT 0 2
            # Measure qubits
            M 0 1 2
            # Check parity (these should always agree)
            DETECTOR rec[-3] rec[-2]  # Z1*Z2
            DETECTOR rec[-3] rec[-1]  # Z0*Z2
        """)
        
        # Create directory structure
        (test_dir / "circuit1").mkdir()
        with open(test_dir / "circuit1" / "circuit_noisy.stim", "w") as f:
            f.write(str(valid_circuit))
        
        # Run simulation
        try:
            sim = ExperimentSimulator(test_dir, shots=10)
            sim.run_simulation()
            print("✔ run_simulation passed")
            
            # Verify results
            samples, cids, table = sim.data_manager.get_results()
            assert len(samples) == 10  # 10 shots
            assert len(table) == 1
            print("✔ results validation passed")
            
            # Test saving
            output_dir = test_dir / "output"
            output_dir.mkdir()
            sim.save_results(output_dir/"results.npz", output_dir/"table.json")
            assert (output_dir/"results.npz").exists()
            print("✔ save_results passed")
            
        except Exception as e:
            print(f"✖ Simulation test failed: {e}")
            raise
            
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
    
    print("=== ExperimentSimulator testing complete ===")

if __name__ == "__main__":
    test_experiment_simulator()