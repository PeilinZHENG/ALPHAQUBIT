from pathlib import Path
import stim
import numpy as np
import shutil
import json

class TestUtils:
    @staticmethod
    def create_test_environment() -> Path:
        """Create a test environment with sample circuits."""
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
        
        return test_dir

    @staticmethod
    def validate_results(samples: np.ndarray, cids: np.ndarray, table: list) -> bool:
        """Validate simulation results."""
        if len(samples) == 0:
            return False
        if len(cids) != len(samples):
            return False
        if len(table) != len(np.unique(cids)):
            return False
        return True

def test_test_utils():
    """Test function for TestUtils"""
    print("\n=== Testing TestUtils ===")
    
    # Test create_test_environment
    test_dir = None
    try:
        test_dir = TestUtils.create_test_environment()
        stim_file = test_dir/"circuit1"/"circuit_noisy.stim"
        if stim_file.exists():
            print("✔ create_test_environment created files")
            
            # Verify circuit file content
            with open(stim_file, 'r') as f:
                content = f.read()
                if "R 0 1" in content and "DETECTOR" in content:
                    print("✔ Circuit file contains expected content")
                else:
                    print("✖ Circuit file missing expected content")
        else:
            print("✖ create_test_environment failed to create files")
    
    finally:
        # Clean up
        if test_dir and test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
    
    # Test validate_results
    test_samples = np.random.randint(0, 2, (10, 5))
    test_cids = np.array([0]*5 + [1]*5, dtype=np.uint16)
    test_table = ["test1.stim", "test2.stim"]
    
    if TestUtils.validate_results(test_samples, test_cids, test_table):
        print("✔ validate_results passed with good data")
    else:
        print("✖ validate_results failed with good data")
    
    # Test empty data
    if not TestUtils.validate_results(np.array([]), np.array([]), []):
        print("✔ validate_results caught empty data")
    else:
        print("✖ validate_results failed with empty data")
    
    # Test mismatched data
    if not TestUtils.validate_results(test_samples, np.array([0]*10), test_table):
        print("✔ validate_results caught mismatched data")
    else:
        print("✖ validate_results failed with mismatched data")

    print("=== TestUtils testing complete ===\n")

if __name__ == "__main__":
    test_test_utils()