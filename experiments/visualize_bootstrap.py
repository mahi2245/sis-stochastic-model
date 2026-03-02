#!/usr/bin/env python3
"""Visualize parametric bootstrap results"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Usage: python visualize_bootstrap.py <bootstrap_results.csv> [true_beta] [true_gamma]")
    sys.exit(1)

# Load results
bootstrap_df = pd.read_csv(sys.argv[1])

# Get true values (from command line or from CSV)
if len(sys.argv) >= 4:
    true_beta = float(sys.argv[2])
    true_gamma = float(sys.argv[3])
else:
    # Use values from first row of CSV
    true_beta = bootstrap_df['true_beta'].iloc[0]
    true_gamma = bootstrap_df['true_gamma'].iloc[0]

true_r0 = true_beta / true_gamma

# Extract estimates
beta_hats = bootstrap_df['beta_hat'].values
gamma_hats = bootstrap_df['gamma_hat'].values
r0_hats = bootstrap_df['r0_hat'].values

# Compute statistics
beta_ci = np.percentile(beta_hats, [2.5, 97.5])
gamma_ci = np.percentile(gamma_hats, [2.5, 97.5])
r0_ci = np.percentile(r0_hats, [2.5, 97.5])

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Parametric Bootstrap Results', fontsize=16, fontweight='bold')

# Beta histogram
ax = axes[0, 0]
ax.hist(beta_hats, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(true_beta, color='red', linestyle='--', linewidth=2, label='True β')
ax.axvline(beta_ci[0], color='green', linestyle=':', linewidth=1.5, label='95% CI')
ax.axvline(beta_ci[1], color='green', linestyle=':', linewidth=1.5)
ax.axvline(np.mean(beta_hats), color='orange', linestyle='-', linewidth=2, label='Mean β̂')
ax.set_xlabel('β', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'β Distribution\nTrue={true_beta:.4f}, CI=[{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]')
ax.legend()
ax.grid(alpha=0.3)

# Gamma histogram
ax = axes[0, 1]
ax.hist(gamma_hats, bins=30, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(true_gamma, color='red', linestyle='--', linewidth=2, label='True γ')
ax.axvline(gamma_ci[0], color='green', linestyle=':', linewidth=1.5, label='95% CI')
ax.axvline(gamma_ci[1], color='green', linestyle=':', linewidth=1.5)
ax.axvline(np.mean(gamma_hats), color='orange', linestyle='-', linewidth=2, label='Mean γ̂')
ax.set_xlabel('γ', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'γ Distribution\nTrue={true_gamma:.4f}, CI=[{gamma_ci[0]:.4f}, {gamma_ci[1]:.4f}]')
ax.legend()
ax.grid(alpha=0.3)

# R0 histogram
ax = axes[1, 0]
ax.hist(r0_hats, bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
ax.axvline(true_r0, color='red', linestyle='--', linewidth=2, label='True R₀')
ax.axvline(r0_ci[0], color='green', linestyle=':', linewidth=1.5, label='95% CI')
ax.axvline(r0_ci[1], color='green', linestyle=':', linewidth=1.5)
ax.axvline(np.mean(r0_hats), color='orange', linestyle='-', linewidth=2, label='Mean R̂₀')
ax.set_xlabel('R₀', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'R₀ Distribution\nTrue={true_r0:.4f}, CI=[{r0_ci[0]:.4f}, {r0_ci[1]:.4f}]')
ax.legend()
ax.grid(alpha=0.3)

# Beta vs Gamma scatter
ax = axes[1, 1]
ax.scatter(beta_hats, gamma_hats, alpha=0.5, s=20, color='purple')
ax.scatter([true_beta], [true_gamma], color='red', s=200, marker='*', 
           edgecolor='black', linewidth=2, label='True (β, γ)', zorder=5)
ax.scatter([np.mean(beta_hats)], [np.mean(gamma_hats)], color='orange', s=150, 
           marker='o', edgecolor='black', linewidth=2, label='Mean (β̂, γ̂)', zorder=5)
ax.set_xlabel('β', fontsize=12)
ax.set_ylabel('γ', fontsize=12)
ax.set_title('β vs γ Correlation')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
output_file = sys.argv[1].replace('.csv', '_plot.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_file}")

# Print detailed summary
print("\n" + "="*70)
print("DETAILED RESULTS")
print("="*70)

def check_coverage(true_val, ci_low, ci_high):
    return "✓ COVERED" if ci_low <= true_val <= ci_high else "✗ NOT COVERED"

print(f"\nβ:")
print(f"  True value:     {true_beta:.6f}")
print(f"  Mean estimate:  {np.mean(beta_hats):.6f}")
print(f"  Std dev:        {np.std(beta_hats, ddof=1):.6f}")
print(f"  95% CI:         [{beta_ci[0]:.6f}, {beta_ci[1]:.6f}]")
print(f"  Coverage:       {check_coverage(true_beta, beta_ci[0], beta_ci[1])}")
print(f"  Bias:           {np.mean(beta_hats) - true_beta:+.6f}")

print(f"\nγ:")
print(f"  True value:     {true_gamma:.6f}")
print(f"  Mean estimate:  {np.mean(gamma_hats):.6f}")
print(f"  Std dev:        {np.std(gamma_hats, ddof=1):.6f}")
print(f"  95% CI:         [{gamma_ci[0]:.6f}, {gamma_ci[1]:.6f}]")
print(f"  Coverage:       {check_coverage(true_gamma, gamma_ci[0], gamma_ci[1])}")
print(f"  Bias:           {np.mean(gamma_hats) - true_gamma:+.6f}")

print(f"\nR₀:")
print(f"  True value:     {true_r0:.6f}")
print(f"  Mean estimate:  {np.mean(r0_hats):.6f}")
print(f"  Std dev:        {np.std(r0_hats, ddof=1):.6f}")
print(f"  95% CI:         [{r0_ci[0]:.6f}, {r0_ci[1]:.6f}]")
print(f"  Coverage:       {check_coverage(true_r0, r0_ci[0], r0_ci[1])}")
print(f"  Bias:           {np.mean(r0_hats) - true_r0:+.6f}")

print(f"\nCorrelation β̂ vs γ̂: {np.corrcoef(beta_hats, gamma_hats)[0,1]:.4f}")
print("="*70)