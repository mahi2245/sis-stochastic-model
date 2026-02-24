#!/usr/bin/env python3
"""Visualize delete-d jackknife results"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Usage: python visualize_jackknife.py <jackknife_results.csv>")
    sys.exit(1)

# Load results
df = pd.read_csv(sys.argv[1])

print("="*70)
print("JACKKNIFE RESULTS SUMMARY")
print("="*70)
print(f"Total predictions: {len(df)}")
print(f"Number of iterations: {df['iteration'].max() + 1}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Delete-d Jackknife Results', fontsize=16, fontweight='bold')

params = ['beta', 'gamma', 'r0']
colors = ['steelblue', 'coral', 'mediumseagreen']

for idx, param in enumerate(params):
    pred_col = f'{param}_pred'
    true_col = f'{param}_true'
    error_col = f'{param}_error'
    
    predictions = df[pred_col].values
    true_values = df[true_col].values
    errors = df[error_col].values
    
    # Row 1: Predicted vs True scatter
    ax = axes[0, idx]
    ax.scatter(true_values, predictions, alpha=0.5, s=30, color=colors[idx])
    
    # Perfect prediction line
    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel(f'True {param}', fontsize=11)
    ax.set_ylabel(f'Predicted {param}', fontsize=11)
    ax.set_title(f'{param.upper()}: Predicted vs True')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add R² to plot
    correlation = np.corrcoef(true_values, predictions)[0, 1]
    r_squared = correlation ** 2
    ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 2: Error distribution
    ax = axes[1, idx]
    ax.hist(errors, bins=30, alpha=0.7, color=colors[idx], edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(np.mean(errors), color='orange', linestyle='-', linewidth=2, label='Mean error')
    
    ax.set_xlabel(f'{param} Error (Predicted - True)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{param.upper()}: Error Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add stats to plot
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

# Print detailed statistics
print("\n" + "="*70)
print("DETAILED STATISTICS")
print("="*70)

for param in params:
    pred_col = f'{param}_pred'
    true_col = f'{param}_true'
    error_col = f'{param}_error'
    
    predictions = df[pred_col].values
    true_values = df[true_col].values
    errors = df[error_col].values
    
    # Statistics
    mean_pred = predictions.mean()
    std_pred = predictions.std(ddof=1)
    mean_true = true_values.mean()
    
    bias = errors.mean()
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())
    
    # Correlation
    correlation = np.corrcoef(true_values, predictions)[0, 1]
    r_squared = correlation ** 2
    
    print(f"\n{param.upper()}:")
    print(f"  Mean prediction:  {mean_pred:.6f}")
    print(f"  Prediction std:   {std_pred:.6f}")
    print(f"  Mean true value:  {mean_true:.6f}")
    print(f"  Bias (avg error): {bias:+.6f}")
    print(f"  MAE:              {mae:.6f}")
    print(f"  RMSE:             {rmse:.6f}")
    print(f"  R²:               {r_squared:.6f}")
    print(f"  Correlation:      {correlation:.6f}")

print("\n" + "="*70)

# Check for systematic bias
print("\nINTERPRETATION:")
print("="*70)
for param in params:
    error_col = f'{param}_error'
    errors = df[error_col].values
    bias = errors.mean()
    mae = np.abs(errors).mean()
    
    if abs(bias) < 0.01:
        bias_msg = "✓ Low bias (unbiased estimates)"
    elif abs(bias) < 0.05:
        bias_msg = "⚠ Moderate bias"
    else:
        bias_msg = "✗ High bias (systematic error)"
    
    print(f"{param.upper():5s}: {bias_msg}")

print("="*70)