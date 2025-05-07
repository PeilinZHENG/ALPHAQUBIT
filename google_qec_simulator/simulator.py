from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import random
import stim
import json
import sys

class ExperimentSimulator:
    def __init__(self, root_dir: Path, shots: int):
        self.root = root_dir.resolve()
        self.shots = shots
        self.blocks: List[np.ndarray] = []
        self.cid_blocks: List[np.ndarray] = []
        self.table: List[str] = []

    def _locate_stim_and_dem(self, folder: Path) -> Tuple[Path, Optional[Path]]:
        stim_path = folder / "circuit_noisy.stim"
        if not stim_path.exists():
            stim_files = list(folder.glob("*.stim"))
            if not stim_files:
                raise FileNotFoundError(f"No .stim files found in {folder}")
            stim_path = stim_files[0]
        
        dem_files = list(folder.glob("*.dem"))
        dem_path = dem_files[0] if dem_files else None
        return stim_path, dem_path

    def _validate_circuit(self, circuit: stim.Circuit) -> None:
        """Check if the circuit has deterministic detectors."""
        try:
            dem = circuit.detector_error_model(ignore_decomposition_failures=True)
        except ValueError as e:
            # Alternative way to save diagram that works with older Stim versions
            debug_file = "debug_detectors.svg"
            with open(debug_file, "w") as f:
                f.write(str(circuit.diagram("detslice-with-ops-svg")))
            print(f"\n⚠️ Non-deterministic detectors detected. Debug SVG saved to {debug_file}")
            raise ValueError(f"Circuit has non-deterministic detectors. See {debug_file}") from e

    def _create_sampler(self, stim_path: Path, dem_path: Optional[Path]):
        circuit = stim.Circuit.from_file(stim_path)
        self._validate_circuit(circuit)
        
        if dem_path:
            dem = stim.DetectorErrorModel.from_file(dem_path)
        else:
            dem = circuit.detector_error_model(ignore_decomposition_failures=True)
        
        return dem.compile_sampler()

    def _sample_circuit(self, stim_path: Path, dem_path: Optional[Path]) -> np.ndarray:
        sampler = self._create_sampler(stim_path, dem_path)
        return sampler.sample(self.shots, seed=random.randrange(2**32)).astype(np.uint8)

    def run_simulation(self):
        dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not dirs:
            raise ValueError(f"No subdirectories found in {self.root}")
        
        success_count = 0
        for cid, folder in enumerate(dirs):
            try:
                stim_path, dem_path = self._locate_stim_and_dem(folder)
                print(f"→ {folder.name}: {self.shots:,} shots (cid={cid})")
                
                samples = self._sample_circuit(stim_path, dem_path)
                self._store_results(cid, stim_path, samples)
                success_count += 1
                
            except Exception as e:
                print(f"  ✖ Failed {folder.name}: {type(e).__name__} - {str(e)}")
                continue

        if success_count == 0:
            raise RuntimeError("All simulations failed. Check circuit definitions.")

    def _store_results(self, cid: int, stim_path: Path, samples: np.ndarray):
        self.table.append(str(stim_path))
        self.blocks.append(samples)
        self.cid_blocks.append(np.full(len(samples), cid, dtype=np.uint16))

    def get_results(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        all_samples = np.concatenate(self.blocks, axis=0)
        all_cids = np.concatenate(self.cid_blocks, axis=0)
        rng = np.random.default_rng()
        order = rng.permutation(len(all_samples))
        return all_samples[order], all_cids[order], self.table

    def save_results(self, npz_path: Path, table_path: Path):
        samples, cids, table = self.get_results()
        np.savez_compressed(npz_path, data=samples, circuit_id=cids)
        with open(table_path, 'w') as f:
            json.dump(table, f, indent=2)
        print(f"\n✔ Saved {npz_path} and {table_path} (rows={len(samples):,})")


def test_simulator():
    """Test function with a sample circuit."""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create a valid test circuit
    circuit = stim.Circuit("""
        R 0 1    # Reset qubits
        H 0      # Hadamard on qubit 0
        CNOT 0 1 # Entangle qubits
        M 0 1    # Measure both qubits
        DETECTOR rec[-1]  # Check consistency
        DETECTOR rec[-2]
    """)
    
    # Save to test directory
    (test_dir / "circuit1").mkdir(exist_ok=True)
    with open(test_dir / "circuit1" / "circuit_noisy.stim", "w") as f:
        f.write(str(circuit))
    
    # Run simulation
    print("Starting simulator test...")
    try:
        sim = ExperimentSimulator(test_dir, shots=100)
        sim.run_simulation()
        
        # Print results
        samples, cids, table = sim.get_results()
        print("\n✔ Test successful!")
        print(f"Total samples: {len(samples)}")
        print(f"Circuit IDs: {np.unique(cids)}")
        print(f"Circuit table: {table}")
        
        # Save results
        test_out = Path("test_output")
        test_out.mkdir(exist_ok=True)
        sim.save_results(test_out / "test_samples.npz", test_out / "test_table.json")
    except Exception as e:
        print(f"\n✖ Test failed: {type(e).__name__} - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    test_simulator()