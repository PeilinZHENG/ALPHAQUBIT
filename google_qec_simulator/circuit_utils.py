from pathlib import Path
from typing import Optional, Tuple, Union
import stim
import numpy as np
import shutil
import sys

class CircuitUtils:
    @staticmethod
    def locate_stim_and_dem(folder: Path) -> Tuple[Path, Optional[Path]]:
        """Tested: Working correctly"""
        print("\n[TEST] locate_stim_and_dem")
        try:
            print(f"Looking in: {folder}")
            stim_path = folder / "circuit_noisy.stim"
            if not stim_path.exists():
                print("Default stim not found, searching...")
                stim_files = list(folder.glob("*.stim"))
                if not stim_files:
                    raise FileNotFoundError(f"No .stim files found in {folder}")
                stim_path = stim_files[0]
            
            print(f"Found stim file: {stim_path}")
            
            dem_files = list(folder.glob("*.dem"))
            dem_path = dem_files[0] if dem_files else None
            print(f"DEM file: {dem_path if dem_path else 'None found'}")
            
            return stim_path, dem_path
        except Exception as e:
            print(f"[ERROR] locate_stim_and_dem: {type(e).__name__}: {e}")
            raise

    @staticmethod
    def validate_circuit(circuit: stim.Circuit) -> None:
        """Tested: Working correctly"""
        print("\n[TEST] validate_circuit")
        try:
            print("Validating circuit...")
            dem = circuit.detector_error_model(ignore_decomposition_failures=True)
            print("✔ Circuit validated successfully")
        except ValueError as e:
            print("❌ Circuit validation failed")
            debug_file = "debug_detectors.svg"
            try:
                svg_content = circuit.diagram("detslice-with-ops-svg")
                if hasattr(svg_content, 'save'):
                    svg_content.save(debug_file)
                else:
                    with open(debug_file, "w") as f:
                        f.write(str(svg_content))
                print(f"Debug SVG saved to {debug_file}")
            except Exception as diagram_error:
                print(f"Failed to save diagram: {diagram_error}")
            raise ValueError(f"Invalid circuit: {e}") from e

    @staticmethod
    def create_sampler(stim_path: Path, dem_path: Optional[Path]):
        """Testing this thoroughly"""
        print("\n[TEST] create_sampler")
        try:
            print(f"Creating sampler from: {stim_path}")
            print(f"Using DEM file: {dem_path if dem_path else 'None'}")
            
            circuit = stim.Circuit.from_file(stim_path)
            print("✔ Circuit loaded successfully")
            
            CircuitUtils.validate_circuit(circuit)
            print("✔ Circuit validated")
            
            if dem_path:
                print("Loading DEM from file...")
                dem = stim.DetectorErrorModel.from_file(dem_path)
            else:
                print("Generating DEM from circuit...")
                dem = circuit.detector_error_model(ignore_decomposition_failures=True)
            
            print("✔ DEM obtained")
            print("Compiling sampler...")
            sampler = dem.compile_sampler()
            print("✔ Sampler compiled successfully")
            
            return sampler
        except Exception as e:
            print(f"[ERROR] create_sampler: {type(e).__name__}: {e}")
            raise

    @staticmethod
    def sample_circuit(sampler, shots: int) -> np.ndarray:
        """Testing return value handling"""
        print("\n[TEST] sample_circuit")
        try:
            print(f"Sampling {shots} shots...")
            result = sampler.sample(shots=shots)
            print(f"Sampling complete. Result type: {type(result)}")
            
            if isinstance(result, tuple):
                print(f"Tuple received with {len(result)} elements")
                print("Element types:", [type(x) for x in result])
                print("Element shapes:", [getattr(x, 'shape', 'no shape') for x in result])
                
                # Handle different Stim versions
                if len(result) == 2:
                    print("Returning detectors only")
                    return result[0]
                else:
                    print("Unexpected tuple length, trying to extract detectors")
                    # Try to find the numpy array containing detectors
                    for item in result:
                        if isinstance(item, np.ndarray):
                            return item
                    raise ValueError(f"Couldn't find detectors in tuple of length {len(result)}")
            
            elif isinstance(result, np.ndarray):
                print("Returning numpy array directly")
                return result
            
            else:
                print("Converting to numpy array")
                return np.array(result)
                
        except Exception as e:
            print(f"[ERROR] sample_circuit: {type(e).__name__}: {e}")
            raise

def test_circuit_utils():
    """Comprehensive test function"""
    print("\n=== Starting CircuitUtils Test ===")
    test_dir = Path("test_circuit_utils_debug")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Create test circuit
        print("\nCreating test circuit...")
        test_circuit = stim.Circuit("""
            R 0 1 2 3
            H 0
            CNOT 0 1
            CNOT 0 2
            CNOT 0 3
            M 0 1 2 3
            DETECTOR rec[-4] rec[-3]
            DETECTOR rec[-4] rec[-2]
            DETECTOR rec[-4] rec[-1]
        """)
        
        circuit_dir = test_dir / "surface_code"
        circuit_dir.mkdir(exist_ok=True)
        circuit_path = circuit_dir / "circuit_noisy.stim"
        
        print(f"Saving circuit to: {circuit_path}")
        with open(circuit_path, "w") as f:
            f.write(str(test_circuit))
        
        # Test 1: File location
        print("\n--- Testing locate_stim_and_dem ---")
        try:
            stim_path, dem_path = CircuitUtils.locate_stim_and_dem(circuit_dir)
            assert stim_path == circuit_path
            print("✔ Test passed: Found correct stim file")
        except Exception as e:
            print(f"✖ Test failed: {e}")
            raise
        
        # Test 2: Circuit validation
        print("\n--- Testing validate_circuit ---")
        try:
            CircuitUtils.validate_circuit(test_circuit)
            print("✔ Test passed: Circuit validated")
        except Exception as e:
            print(f"✖ Test failed: {e}")
            raise
        
        # Test 3: Sampler creation and sampling
        print("\n--- Testing create_sampler and sample_circuit ---")
        try:
            sampler = CircuitUtils.create_sampler(circuit_path, None)
            print("\nSampler created successfully, now testing sampling...")
            
            samples = CircuitUtils.sample_circuit(sampler, shots=5)
            print(f"\nSample results: {type(samples)}, shape: {getattr(samples, 'shape', 'no shape')}")
            
            if isinstance(samples, np.ndarray):
                print(f"Sample data:\n{samples}")
                print("✔ Test passed: Sampling successful")
            else:
                raise TypeError(f"Expected numpy array, got {type(samples)}")
                
        except Exception as e:
            print(f"✖ Test failed: {e}")
            raise
            
    finally:
        print("\nCleaning up...")
        shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n=== All CircuitUtils Tests Completed ===")

if __name__ == "__main__":
    try:
        test_circuit_utils()
    except Exception as e:
        print(f"\n⚠️ Critical test failure: {type(e).__name__}: {e}")
        sys.exit(1)