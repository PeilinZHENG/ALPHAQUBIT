import argparse
from pathlib import Path
from experiment_simulator import ExperimentSimulator
import stim
import shutil
import sys

def create_valid_test_environment(test_dir: Path) -> Path:
    """Create a test environment with valid surface code circuits"""
    test_dir.mkdir(exist_ok=True)
    
    # Create valid surface code circuit
    surface_code = stim.Circuit("""
        # Surface code with deterministic detectors
        R 0 1 2 3
        H 0
        CNOT 0 1
        CNOT 0 2
        CNOT 0 3
        M 0 1 2 3
        DETECTOR rec[-4] rec[-3]  # Z1*Z2
        DETECTOR rec[-4] rec[-2]  # Z0*Z2
        DETECTOR rec[-4] rec[-1]  # Z0*Z1
    """)
    
    (test_dir / "surface_code").mkdir(exist_ok=True)
    with open(test_dir / "surface_code" / "circuit_noisy.stim", "w") as f:
        f.write(str(surface_code))
    
    return test_dir

def process_experiment_data(data_dir: Path, output_dir: Path, shots: int):
    """Process all subfolders (experiment datasets) in the given directory."""
    # Create test environment if none exists
    if not data_dir.exists():
        print("⚠️ No data found - creating test environment...")
        data_dir = create_valid_test_environment(data_dir)
    
    try:
        # Initialize simulator
        sim = ExperimentSimulator(data_dir, shots=shots)
        print(f"\nFound {len(list(data_dir.iterdir()))} circuit directories")
        
        # Run simulation
        print("\nRunning simulation for all experiments...")
        sim.run_simulation()
        
        # Save results
        output_dir.mkdir(exist_ok=True)
        sim.save_results(output_dir/"samples.npz", output_dir/"circuit_table.json")
        
        print("\n✔ Simulation completed successfully for all experiments!")
        
    except Exception as e:
        print(f"\n✖ Error: {type(e).__name__} - {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Optional: Clean up test data if desired
        pass

def main():
    parser = argparse.ArgumentParser(description="Quantum Error Correction Simulator")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        help="Path to the directory containing experiment subfolders (datasets)"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="results", 
        help="Path to save the simulation results (default: 'results')"
    )
    parser.add_argument(
        '--shots', 
        type=int, 
        default=1000, 
        help="Number of shots for each simulation (default: 1000)"
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Convert data_dir and output_dir to Path objects
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    print(f"Using data directory: {data_dir}")
    print(f"Results will be saved to: {output_dir}")
    
    process_experiment_data(data_dir, output_dir, args.shots)

if __name__ == "__main__":
    main()
