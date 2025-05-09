import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def load_results(results_dir="."):
    """Improved loading with validation"""
    results_dir = Path(results_dir)
    try:
        data = np.load(results_dir / "samples.npz")
        with open(results_dir / "circuit_table.json") as f:
            circuits = json.load(f)
        
        if len(data['data']) == 0:
            raise ValueError("No samples found in data file")
            
        return {
            'samples': data['data'],
            'circuit_ids': data['circuit_id'],
            'circuits': circuits
        }
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        print(f"Files in directory: {list(results_dir.glob('*'))}")
        raise

def analyze_samples(samples):
    """Enhanced analysis with more metrics"""
    stats = {
        'detector_errors': samples.mean(axis=0),
        'logical_error': samples.any(axis=1).mean(),
        'shot_counts': len(samples),
        'no_error_shots': (~samples.any(axis=1)).sum(),
        'correlation_matrix': np.corrcoef(samples.T) if samples.shape[1] > 1 else None
    }
    return stats

def generate_report(analysis):
    """More detailed reporting"""
    report = [
        "Enhanced QEC Analysis Report",
        "="*50,
        f"Total samples analyzed: {analysis['shot_counts']}",
        f"Perfect shots (no errors): {analysis['no_error_shots']} ({analysis['no_error_shots']/analysis['shot_counts']:.1%})",
        f"Logical error rate: {analysis['logical_error']:.4f}",
        "\nDetector Statistics:"
    ]
    
    for i, rate in enumerate(analysis['detector_errors']):
        report.append(f"D{i}: Error rate = {rate:.4f}")
    
    if analysis['correlation_matrix'] is not None:
        report.append("\nDetector Correlations:")
        for i in range(analysis['correlation_matrix'].shape[0]):
            for j in range(i+1, analysis['correlation_matrix'].shape[1]):
                report.append(f"D{i}-D{j}: {analysis['correlation_matrix'][i,j]:.3f}")
    
    return "\n".join(report)

def plot_enhanced_results(samples, save_path=None):
    """More informative visualization"""
    plt.figure(figsize=(15, 5))
    
    # Error distribution plot
    plt.subplot(1, 3, 1)
    error_counts = samples.sum(axis=1)
    plt.hist(error_counts, bins=range(error_counts.max()+2))
    plt.title('Errors per Shot Distribution')
    plt.xlabel('Number of Errors')
    plt.ylabel('Frequency')
    
    # Detector error rates
    plt.subplot(1, 3, 2)
    plt.bar(range(samples.shape[1]), samples.mean(axis=0))
    plt.title('Detector Error Rates')
    plt.xlabel('Detector Index')
    plt.ylabel('Error Rate')
    plt.ylim(0, 1)
    
    # Error correlation heatmap
    if samples.shape[1] > 1:
        plt.subplot(1, 3, 3)
        corr = np.corrcoef(samples.T)
        plt.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.title('Detector Correlation')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def main():
    print("Running enhanced QEC results analysis...")
    
    try:
        # Load data
        data = load_results()
        samples = data['samples']
        
        # Analyze
        analysis = analyze_samples(samples)
        
        # Generate outputs
        print("\n" + generate_report(analysis))
        plot_enhanced_results(samples, "enhanced_analysis.png")
        
        # Special case handling
        if analysis['logical_error'] == 0:
            print("\n🔍 Perfect results detected - possible scenarios:")
            print("1. No noise model was included in the simulation")
            print("2. Error correction worked perfectly (unlikely at scale)")
            print("3. Test circuit was too simple")
            print("\nRecommendation: Add noise to your circuit or try more complex tests")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()