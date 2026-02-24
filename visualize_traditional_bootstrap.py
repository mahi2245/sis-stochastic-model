#!/usr/bin/env python3
"""Visualize traditional bootstrap results"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Usage: python visualize_traditional_bootstrap.py <results.csv>")
    sys.exit(1)

# Load results
df = pd.read_csv(sys.argv[1])
n_samples = len(df)

print("="*70)
print("TRADITIONAL BOOTSTRAP VISUALIZATION")
print("="*70)
print(f"Test samples: {n_samples}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Traditional Bootstrap Results (Test Set)', fontsize=16, fontweight='bold')

params = ['beta', 'gamma', 'r0']
colors = ['steelblue', 'coral', 'mediumseagreen']

for idx, param in enumerate(params):
    true_col = f'{param}_true'
    mean_col = f'{param}_mean'
    ci_low_col = f'{param}_ci_low'
    ci_high_col = f'{param}_ci_high'
    covered_col = f'{param}_covered'
    
    true_vals = df[true_col].values
    mean_vals = df[mean_col].values
    ci_low = df[ci_low_col].values
    ci_high = df[ci_high_col].values
    covered = df[covered_col].values
    
    # Top row: Predictions with error bars
    ax = axes[0, idx]
    
    # Sort by true value for better visualization
    sort_idx = np.argsort(true_vals)
    x_pos = np.arange(n_samples)
    
    # Plot predictions with CIs
    for i, si in enumerate(sort_idx):
        color = 'green' if covered[si] else 'red'
        ax.plot([i, i], [ci_low[si], ci_high[si]], color=color, alpha=0.3, linewidth=1)
    
    ax.scatter(x_pos, mean_vals[sort_idx], color=colors[idx], s=30, alpha=0.7, 
               label='Predicted mean', zorder=3)
    ax.scatter(x_pos, true_vals[sort_idx], color='red', s=20, alpha=0.8, 
               marker='x', label='True value', zorder=4)
    
    ax.set_xlabel('Test Sample (sorted by true value)', fontsize=11)
    ax.set_ylabel(f'{param}', fontsize=11)
    ax.set_title(f'{param.upper()}: Predictions with 95% CI')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add coverage info
    coverage = covered.mean() * 100
    ax.text(0.02, 0.98, f'Coverage: {coverage:.1f}%', transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Bottom row: Error distribution
    ax = axes[1, idx]
    errors = mean_vals - true_vals
    
    ax.hist(errors, bins=20, alpha=0.7, color=colors[idx], edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(np.mean(errors), color='orange', linestyle='-', linewidth=2, 
               label='Mean error')
    
    ax.set_xlabel(f'{param} Error (Predicted - True)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{param.upper()}: Error Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add stats
    bias = np.mean(errors)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    stats_text = f'Bias: {bias:+.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}'
    ax.text(0.70, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
output_file = sys.argv[1].replace('.csv', '_plot.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: {output_file}")

# Detailed statistics
print("\n" + "="*70)
print("DETAILED STATISTICS")
print("="*70)

for param in params:
    true_col = f'{param}_true'
    mean_col = f'{param}_mean'
    ci_low_col = f'{param}_ci_low'
    ci_high_col = f'{param}_ci_high'
    covered_col = f'{param}_covered'
    std_col = f'{param}_std'
    
    true_vals = df[true_col].values
    mean_vals = df[mean_col].values
    covered = df[covered_col].values
    ci_widths = df[ci_high_col].values - df[ci_low_col].values
    
    errors = mean_vals - true_vals
    coverage = covered.mean() * 100
    bias = errors.mean()
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())
    mean_ci_width = ci_widths.mean()
    mean_uncertainty = df[std_col].mean()
    
    print(f"\n{param.upper()}:")
    print(f"  Coverage (95% CI):    {coverage:.1f}%")
    print(f"  Bias:                 {bias:+.6f}")
    print(f"  MAE:                  {mae:.6f}")
    print(f"  RMSE:                 {rmse:.6f}")
    print(f"  Mean CI width:        {mean_ci_width:.6f}")
    print(f"  Mean bootstrap std:   {mean_uncertainty:.6f}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

# Overall assessment
beta_cov = df['beta_covered'].mean() * 100
gamma_cov = df['gamma_covered'].mean() * 100
r0_cov = df['r0_covered'].mean() * 100

print("\nCoverage Assessment (Target: ~95%):")
print(f"  β:  {beta_cov:.1f}% {'✓' if 90 <= beta_cov <= 100 else '✗'}")
print(f"  γ:  {gamma_cov:.1f}% {'✓' if 90 <= gamma_cov <= 100 else '✗'}")
print(f"  R₀: {r0_cov:.1f}% {'✓' if 90 <= r0_cov <= 100 else '✗'}")

print("\nNote: Green error bars = CI covers true value")
print("      Red error bars = CI does NOT cover true value")
print("="*70)