import os
import subprocess
import sys

def run_simulations_on_subfolders(root_dir, shots=10000):
    """
    Run the QEC simulation on all subfolders of the given directory
    
    Args:
        root_dir (str): Root directory containing experiment data folders
        shots (int): Number of shots to run for each simulation
    """
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory", file=sys.stderr)
        sys.exit(1)
        
    # Get all subfolders
    subfolders = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            full_path = os.path.abspath(os.path.join(dirpath, dirname))
            subfolders.append(full_path)
    
    if not subfolders:
        print(f"No subfolders found in {root_dir}")
        return
    
    print(f"Found {len(subfolders)} subfolders. Running simulations...")
    
    # Run simulation for each subfolder
    for i, folder in enumerate(subfolders, 1):
        cmd = [
            "python",
            r".\google_qec_simulator\main.py",
            folder,
            "--shots",
            str(shots)
        ]
        
        print(f"\n=== Running simulation {i}/{len(subfolders)} ===")
        print(f"Folder: {folder}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Simulation completed successfully")
            print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Simulation failed!")
            print("Error:", e.stderr)
            continue

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <experiment_data_directory>", file=sys.stderr)
        sys.exit(1)
        
    experiment_dir = sys.argv[1]
    run_simulations_on_subfolders(experiment_dir)