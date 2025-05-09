# google_qec_simulator/data_helpers.py
import numpy as np
from pathlib import Path
from stim_helpers import extract_rounds_and_dets


def reshape_detectors(det_flat: np.ndarray, stim_path: Path, shots: int) -> np.ndarray:
    """
    Reshapes the flat detector array into a (shots, rounds, dets_per_round, 1) shape.
    """
    # Get rounds and detectors per round
    R, S = extract_rounds_and_dets(stim_path)
    
    # Debugging: Print out rounds, detectors, and shot information
    print(f"Shots (N): {shots}")
    print(f"Extracted rounds (R): {R}, Detectors per round (S): {S}")
    
    # Calculate the expected reshape size
    expected_size = shots * R * S
    print(f"Expected reshape size (shots * rounds * detectors_per_round): {expected_size}")
    
    # Ensure the reshaping size matches
    print(f"Actual detected data size: {det_flat.size}")
    if det_flat.size != expected_size:
        raise ValueError(f"Cannot reshape {det_flat.size} elements into "
                         f"shape ({shots}, {R}, {S}, 1). "
                         f"Expected size: {expected_size}. Please check the dimensions of your data.")

    return det_flat.reshape(shots, R, S, 1).astype(np.float32)


def iq_sample(state, snr, t):
    mu = state
    z = np.random.normal(mu, 1 / np.sqrt(snr))
    if state >= 1:
        z -= t * np.random.exponential(1.0)
    return z


def pdfs(z, snr, t):
    p0 = np.exp(-snr * z**2)
    p1 = 0.5 * np.exp(-snr * (z - 1) ** 2) + 0.5 * np.exp(-snr * z**2) * np.exp(-z / t) * (z > 0)
    p2 = 0.5 * np.exp(-snr * (z - 2) ** 2) + 0.5 * np.exp(-snr * (z - 1) ** 2) * np.exp(-(z - 1) / (2 * t)) * (z > 1)
    return p0, p1, p2


def post(z, snr, t, priors=(0.495, 0.495, 0.01)):
    w0, w1, w2 = priors
    p0, p1, p2 = pdfs(z, snr, t)
    norm = w0 * p0 + w1 * p1 + w2 * p2
    if norm == 0:
        return 0.5, 0.0
    return (w1 * p1) / norm, (w2 * p2) / norm


def soft_channels(n, snr=10.0, t=0.01, leak_p=0.00275):
    states = np.random.choice([0, 1, 2], size=n, p=[1 - leak_p - 0.5, 0.5, leak_p])
    zvals = np.vectorize(iq_sample)(states, snr, t)
    post1, post2 = np.vectorize(post)(zvals, snr, t)
    return post1.astype(np.float32), post2.astype(np.float32)






# Test each function in this file
if __name__ == "__main__":
    stim_path = Path("path_to_a_stim_file/stim_example.stim")

    # Test reshape_detectors
    det_flat = np.random.randint(0, 2, (1000, 100))  # Fake data
    reshaped_det = reshape_detectors(det_flat, stim_path)
    print(f"Reshaped detector shape: {reshaped_det.shape}")

    # Test soft channels generation
    post1, post2 = soft_channels(1000)
    print(f"Generated post1 and post2 channels shapes: {post1.shape}, {post2.shape}")
