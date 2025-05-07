from pathlib import Path
from typing import List, Tuple
import numpy as np
import json
import shutil

class DataManager:
    def __init__(self):
        self.blocks: List[np.ndarray] = []
        self.cid_blocks: List[np.ndarray] = []
        self.table: List[str] = []

    def store_results(self, cid: int, stim_path: Path, samples: np.ndarray) -> None:
        """Store simulation results in memory."""
        print(f"Storing results for {stim_path}, samples shape: {samples.shape}")
        
        # Check for shape consistency
        if self.blocks:
            target_shape = self.blocks[0].shape  # Use the shape of the first sample as target
            if samples.shape != target_shape:
                print(f"Padding samples from shape {samples.shape} to target shape {target_shape}")
                pad_width = [(0, target_shape[i] - samples.shape[i]) for i in range(len(samples.shape))]
                samples = np.pad(samples, pad_width, mode='constant', constant_values=0)
        
        self.table.append(str(stim_path))
        self.blocks.append(samples)
        self.cid_blocks.append(np.full(len(samples), cid, dtype=np.uint16))

    def get_results(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get all simulation results with random shuffling."""
        # Concatenate all the samples from different experiments
        all_samples = np.concatenate(self.blocks, axis=0)
        all_cids = np.concatenate(self.cid_blocks, axis=0)
        rng = np.random.default_rng()  # Random number generator
        order = rng.permutation(len(all_samples))  # Random shuffle order
        return all_samples[order], all_cids[order], self.table

    def save_results(self, npz_path: Path, table_path: Path) -> None:
        """Save results to files."""
        # Get the shuffled results
        samples, cids, table = self.get_results()
        
        # Save the simulation results as npz (compressed)
        np.savez_compressed(npz_path, data=samples, circuit_id=cids)
        
        # Save the circuit names to a JSON file
        with open(table_path, 'w') as f:
            json.dump(table, f, indent=2)

def test_data_manager():
    """Test function for DataManager"""
    print("\n=== Testing DataManager ===")
    
    # Setup test directory
    test_dir = Path("test_data_manager")
    test_dir.mkdir(exist_ok=True)
    
    try:
        dm = DataManager()
        
        # Create test data
        test_samples = np.random.randint(0, 2, (10, 5), dtype=np.uint8)
        
        # Test store_results
        dm.store_results(0, Path("test1.stim"), test_samples)
        dm.store_results(1, Path("test2.stim"), test_samples)
        print("✔ store_results completed")

        # Test get_results
        samples, cids, table = dm.get_results()
        if len(samples) == 20 and len(cids) == 20 and len(table) == 2:
            print("✔ get_results returned correct data")
        else:
            print("✖ get_results returned incorrect data")

        # Test save_results
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)
        dm.save_results(output_dir/"test.npz", output_dir/"test.json")
        
        if (output_dir/"test.npz").exists() and (output_dir/"test.json").exists():
            print("✔ save_results created files successfully")
            
            # Verify NPZ file contents
            with np.load(output_dir/"test.npz") as data:
                if len(data['data']) == 20 and len(data['circuit_id']) == 20:
                    print("✔ NPZ file contains correct data")
                else:
                    print("✖ NPZ file contains incorrect data")
        else:
            print("✖ save_results failed to create files")

    finally:
        # Clean up
        shutil.rmtree(test_dir, ignore_errors=True)

    print("=== DataManager testing complete ===\n")

if __name__ == "__main__":
    test_data_manager()
