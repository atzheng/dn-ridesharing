import pandas as pd
import numpy as np
import glob
import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_metrics(estimator_mean, true_ate):
    """Calculate normalized bias and RMSE."""
    bias = (estimator_mean - true_ate) / true_ate
    rmse = np.sqrt((estimator_mean - true_ate)**2) / true_ate
    return bias, rmse

def analyze_results_file(file_path):
    """Analyze a single results file and return statistics."""
    logging.info(f"Analyzing file: {file_path}")
    df = pd.read_csv(file_path)
    
    # Calculate statistics for each estimator
    stats = {}
    
    # Calculate ATE from A and B columns
    if 'A' in df.columns and 'B' in df.columns:
        ate = df['B'] - df['A']  # Treatment effect is B - A
        stats['true_ate'] = {
            'mean': ate.mean(),
            'std': ate.std(),
            'count': len(ate)
        }
    
    # Calculate statistics for other estimators
    for column in df.columns:
        if column not in ['A', 'B']:  # Skip the raw A/B columns
            stats[column] = {
                'mean': df[column].mean(),
                'std': df[column].std(),
                'count': len(df[column])
            }
    
    return stats

def plot_metrics(summary_df, true_ate, output_dir):
    """Create plots for SD, normalized bias, and normalized RMSE."""
    # Filter out true_ate from the data
    plot_df = summary_df[summary_df['estimator'] != 'true_ate'].copy()
    
    # Convert switch_every to numeric
    plot_df['switch_every'] = pd.to_numeric(plot_df['switch_every'])
    
    # Calculate normalized metrics
    plot_df['bias'] = (plot_df['mean'] - true_ate) / true_ate
    plot_df['rmse'] = np.sqrt((plot_df['mean'] - true_ate)**2) / true_ate
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot 1: Standard Deviation
    sns.lineplot(data=plot_df, x='switch_every', y='std', hue='estimator', marker='o', ax=ax1)
    ax1.set_xscale('log')
    ax1.set_xlabel('Switch Every (seconds)')
    ax1.set_ylabel('Standard Deviation')
    ax1.set_title('Standard Deviation by Switch Duration')
    ax1.grid(True)
    
    # Plot 2: Normalized Bias
    sns.lineplot(data=plot_df, x='switch_every', y='bias', hue='estimator', marker='o', ax=ax2)
    ax2.set_xscale('log')
    ax2.set_xlabel('Switch Every (seconds)')
    ax2.set_ylabel('Normalized Bias')
    ax2.set_title('Normalized Bias by Switch Duration')
    ax2.grid(True)
    
    # Plot 3: Normalized RMSE
    sns.lineplot(data=plot_df, x='switch_every', y='rmse', hue='estimator', marker='o', ax=ax3)
    ax3.set_xscale('log')
    ax3.set_xlabel('Switch Every (seconds)')
    ax3.set_ylabel('Normalized RMSE')
    ax3.set_title('Normalized RMSE by Switch Duration')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_plot.png')
    plt.close()

def main():
    # Get all results files
    results_files = glob.glob('output/switch=*/results.csv')
    ate_file = 'output/ate.csv'
    
    # Create output directory for analysis
    analysis_dir = Path('output/analysis')
    analysis_dir.mkdir(exist_ok=True)
    
    # Analyze each results file
    all_stats = {}
    for file_path in results_files:
        switch_every = file_path.split('switch=')[1].split('/')[0]
        logging.info(f"Processing switch_every={switch_every}")
        stats = analyze_results_file(file_path)
        all_stats[switch_every] = stats
    
    # Analyze ATE file if it exists
    if os.path.exists(ate_file):
        logging.info("Processing ATE file")
        ate_stats = analyze_results_file(ate_file)
        all_stats['ate'] = ate_stats
    
    # Create summary DataFrame
    summary_rows = []
    for switch_every, stats in all_stats.items():
        for estimator, values in stats.items():
            summary_rows.append({
                'switch_every': switch_every,
                'estimator': estimator,
                'mean': values['mean'],
                'std': values['std'],
                'count': values['count']
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary to CSV
    output_file = analysis_dir / 'summary.csv'
    summary_df.to_csv(output_file, index=False)
    logging.info(f"Summary saved to {output_file}")
    
    # Get true ATE value
    true_ate = summary_df[summary_df['estimator'] == 'true_ate']['mean'].iloc[0]
    
    # Create plots
    plot_metrics(summary_df, true_ate, analysis_dir)
    logging.info("Plots saved to output/analysis/metrics_plot.png")
    
    # Print summary
    print("\nResults Summary:")
    print("===============")
    for switch_every in sorted(all_stats.keys()):
        print(f"\nSwitch Every: {switch_every}")
        print("-" * 50)
        for estimator, values in all_stats[switch_every].items():
            print(f"{estimator}:")
            print(f"  Mean: {values['mean']:.4f}")
            print(f"  Std:  {values['std']:.4f}")
            print(f"  N:    {values['count']}")

if __name__ == "__main__":
    main() 