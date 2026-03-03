#!/usr/bin/env python3
"""
visualize_traditional_bootstrap.py
-----------------------------------
Visualize results from traditional_bootstrap.py.

Produces a 2x3 figure:
  - Top row:    per-test-sample predictions with 95% CI error bars (sorted
                by true value), green = CI covers true value, red = does not.
  - Bottom row: prediction error distributions with bias/MAE/RMSE annotations.

Also prints a detailed numerical summary and coverage assessment to stdout.

Usage (from project root):
    python experiments/visualize_traditional_bootstrap.py \\
        results/traditional_bootstrap/traditional_bootstrap_results.csv

Output plot saved to same directory as input CSV, with '_plot.png' suffix.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python experiments/visualize_traditional_bootstrap.py <results.csv>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
n_samples = len(df)

print("=" * 70)
print("TRADITIONAL BOOTSTRAP VISUALIZATION")
print("=" * 70)
print(f"Test samples: {n_samples}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Traditional Bootstrap Results (Test Set)", fontsize=16, fontweight="bold")

params = ["beta", "gamma", "r0"]
colors = ["steelblue", "coral", "mediumseagreen"]

for idx, param in enumerate(params):
    true_vals = df[f"{param}_true"].values
    mean_vals = df[f"{param}_mean"].values
    ci_low    = df[f"{param}_ci_low"].values
    ci_high   = df[f"{param}_ci_high"].values
    covered   = df[f"{param}_covered"].values

    # --- Top row: predictions with CI error bars, sorted by true value ---
    ax = axes[0, idx]
    sort_idx = np.argsort(true_vals)
    x_pos    = np.arange(n_samples)

    for i, si in enumerate(sort_idx):
        color = "green" if covered[si] else "red"
        ax.plot([i, i], [ci_low[si], ci_high[si]], color=color, alpha=0.3, linewidth=1)

    ax.scatter(x_pos, mean_vals[sort_idx], color=colors[idx], s=30, alpha=0.7,
               label="Predicted mean", zorder=3)
    ax.scatter(x_pos, true_vals[sort_idx], color="red", s=20, alpha=0.8,
               marker="x", label="True value", zorder=4)

    coverage = covered.mean() * 100
    ax.set_xlabel("Test Sample (sorted by true value)", fontsize=11)
    ax.set_ylabel(param, fontsize=11)
    ax.set_title(f"{param.upper()}: Predictions with 95% CI")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.98, f"Coverage: {coverage:.1f}%", transform=ax.transAxes,
            verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    # --- Bottom row: error distribution ---
    ax = axes[1, idx]
    errors = mean_vals - true_vals

    ax.hist(errors, bins=20, alpha=0.7, color=colors[idx], edgecolor="black")
    ax.axvline(0,               color="red",    linestyle="--", linewidth=2, label="Zero error")
    ax.axvline(np.mean(errors), color="orange", linestyle="-",  linewidth=2, label="Mean error")

    bias = np.mean(errors)
    mae  = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    ax.set_xlabel(f"{param} Error (Predicted - True)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"{param.upper()}: Error Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(0.70, 0.95,
            f"Bias: {bias:+.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}",
            transform=ax.transAxes, verticalalignment="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))

plt.tight_layout()
output_file = sys.argv[1].replace(".csv", "_plot.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"\nSaved: {output_file}")

# ---------------------------------------------------------------------------
# Numerical summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("DETAILED STATISTICS")
print("=" * 70)

for param in params:
    true_vals  = df[f"{param}_true"].values
    mean_vals  = df[f"{param}_mean"].values
    covered    = df[f"{param}_covered"].values
    ci_widths  = df[f"{param}_ci_high"].values - df[f"{param}_ci_low"].values
    errors     = mean_vals - true_vals

    print(f"\n{param.upper()}:")
    print(f"  Coverage (95% CI):    {covered.mean() * 100:.1f}%")
    print(f"  Bias:                 {errors.mean():+.6f}")
    print(f"  MAE:                  {np.abs(errors).mean():.6f}")
    print(f"  RMSE:                 {np.sqrt((errors**2).mean()):.6f}")
    print(f"  Mean CI width:        {ci_widths.mean():.6f}")
    print(f"  Mean bootstrap std:   {df[f'{param}_std'].mean():.6f}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("\nCoverage Assessment (Target: ~95%):")

for param in params:
    cov = df[f"{param}_covered"].mean() * 100
    mark = "✓" if 90 <= cov <= 100 else "✗"
    print(f"  {param.upper():5s}: {cov:.1f}% {mark}")

print("\nNote: Green error bars = CI covers true value")
print("      Red error bars   = CI does NOT cover true value")
print("=" * 70)