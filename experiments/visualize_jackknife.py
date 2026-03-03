#!/usr/bin/env python3
"""
visualize_jackknife.py
-----------------------
Visualize results from jackknife.py.

Produces a 2x3 figure:
  - Top row:    predicted vs true scatter plots with a perfect-prediction
                reference line and R² annotation.
  - Bottom row: prediction error distributions with bias/MAE/RMSE annotations.

Also prints a detailed numerical summary and bias interpretation to stdout.

Usage (from project root):
    python experiments/visualize_jackknife.py results/jackknife/jackknife_results.csv

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
    print("Usage: python experiments/visualize_jackknife.py <jackknife_results.csv>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])

print("=" * 70)
print("JACKKNIFE RESULTS SUMMARY")
print("=" * 70)
print(f"Total predictions:     {len(df)}")
print(f"Number of iterations:  {df['iteration'].max() + 1}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Delete-d Jackknife Results", fontsize=16, fontweight="bold")

params = ["beta", "gamma", "r0"]
colors = ["steelblue", "coral", "mediumseagreen"]

for idx, param in enumerate(params):
    predictions  = df[f"{param}_pred"].values
    true_values  = df[f"{param}_true"].values
    errors       = df[f"{param}_error"].values

    # --- Top row: predicted vs true ---
    ax = axes[0, idx]
    ax.scatter(true_values, predictions, alpha=0.5, s=30, color=colors[idx])

    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2,
            label="Perfect prediction")

    r_squared = np.corrcoef(true_values, predictions)[0, 1] ** 2
    ax.set_xlabel(f"True {param}", fontsize=11)
    ax.set_ylabel(f"Predicted {param}", fontsize=11)
    ax.set_title(f"{param.upper()}: Predicted vs True")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, f"R² = {r_squared:.4f}", transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # --- Bottom row: error distribution ---
    ax = axes[1, idx]
    ax.hist(errors, bins=30, alpha=0.7, color=colors[idx], edgecolor="black")
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
    predictions = df[f"{param}_pred"].values
    true_values = df[f"{param}_true"].values
    errors      = df[f"{param}_error"].values

    bias        = errors.mean()
    mae         = np.abs(errors).mean()
    rmse        = np.sqrt((errors ** 2).mean())
    r_squared   = np.corrcoef(true_values, predictions)[0, 1] ** 2
    correlation = np.corrcoef(true_values, predictions)[0, 1]

    print(f"\n{param.upper()}:")
    print(f"  Mean prediction:  {predictions.mean():.6f}")
    print(f"  Prediction std:   {predictions.std(ddof=1):.6f}")
    print(f"  Mean true value:  {true_values.mean():.6f}")
    print(f"  Bias (avg error): {bias:+.6f}")
    print(f"  MAE:              {mae:.6f}")
    print(f"  RMSE:             {rmse:.6f}")
    print(f"  R²:               {r_squared:.6f}")
    print(f"  Correlation:      {correlation:.6f}")

# --- Bias interpretation ---
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

for param in params:
    bias = df[f"{param}_error"].values.mean()
    if   abs(bias) < 0.01: msg = "✓ Low bias (unbiased estimates)"
    elif abs(bias) < 0.05: msg = "⚠ Moderate bias"
    else:                  msg = "✗ High bias (systematic error)"
    print(f"{param.upper():5s}: {msg}")

print("=" * 70)