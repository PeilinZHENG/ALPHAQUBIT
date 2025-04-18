# stats.py
import numpy as np
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python stats.py <path_to_npy_file>")
        return

    file_path = sys.argv[1]
    data = np.load(file_path)
    
    # Basic stats
    print(f"\n{' File Info ':-^40}")
    print(f"Path: {file_path}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min/Mean/Max: {data.min()} / {data.mean():.2f} / {data.max()}")
    
    # Display first 100 entries with smart formatting
    print(f"\n{' First 100 Entries ':-^40}")
    flat_data = data.reshape(-1)[:100]  # Handle any dimensionality
    
    with np.printoptions(
        threshold=100, 
        edgeitems=5, 
        linewidth=120,
        formatter={'int': lambda x: f"{x:2d}"}  # Formatting for binary data
    ):
        if len(data.shape) == 1:
            print(flat_data)
        else:
            # For 2D+ data, show first 100 elements with original structure
            print(data[:100] if len(data) > 100 else data)

if __name__ == "__main__":
    main()