#!/usr/bin/env python3
"""
bootstrap_parametric.py
-----------------------
Parametric bootstrap for uncertainty quantification of SISNet predictions.

For a single epidemic observation (fixed beta, gamma, pop, gens), generates B
independent stochastic SIS replicates under those true parameters, feeds each
through the trained SISNet, and collects the resulting distribution of
beta_hat, gamma_hat, and R0_hat. The spread of that distribution is the
bootstrap estimate of prediction uncertainty.

Usage (from project root):
    python experiments/bootstrap_parametric.py \\
        --beta 0.4 --gamma 0.2 --pop 5000 --gens 200 [--B 500] [--seed 123]

Reads:
    models/sis_model_best.pth

Writes:
    results/bootstrap/bootstrap_results.csv
"""

import argparse
import csv
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# src/ is not on the path when running from the project root, so add it.
sys.path.insert(0, "src")

from simple import simulate_sis

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH  = "models/sis_model_best.pth"
OUTPUT_PATH = "results/bootstrap/bootstrap_results.csv"

# Must match TIME_POINTS order used in the SIS training data generation
TIME_POINTS = [0.25, 0.75, 0.10, 0.60, 0.50, 1.0]


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class SISNet(nn.Module):
    """
    SISNet: 8 -> 32 -> 32 -> 16 -> 2

    Input features (8):
        gen_norm, pop_norm, infected_t25, infected_t75,
        infected_t10, infected_t60, infected_t50, infected_t100
        (all infected counts normalized as proportions of population)

    Output (2):
        beta_scaled, gamma_scaled — both in (0, 1); unscale via checkpoint
        min/max values to recover true beta and gamma.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_features_from_subsims(beta, gamma, population, i0, generations, seed=None):
    """
    Mirror the SIS data-generation logic: for each fractional timepoint tp,
    run a fresh simulate_sis up to floor(tp * generations) and record the
    infected count from the final generation of that sub-simulation.

    Parameters
    ----------
    beta, gamma : float
        True epidemic parameters driving each sub-simulation.
    population : int
        Total population size N.
    i0 : int
        Initial infected count.
    generations : int
        Total number of generations in the epidemic.
    seed : int, optional
        NumPy random seed (varied per replicate for independent draws).

    Returns
    -------
    dict
        Keys: gens, pop, i25, i75, i10, i60, i50, i100
    """
    if seed is not None:
        np.random.seed(seed)

    infected_vals = []
    for tp in TIME_POINTS:
        t = max(1, int(tp * generations))
        hist_tp, _ = simulate_sis(
            n=population, m=i0,
            beta=beta, gamma=gamma,
            generations=t, x=0,
        )
        infected_vals.append(int(np.sum(hist_tp[-1])))

    # Unpack in TIME_POINTS order: [t25, t75, t10, t60, t50, t100]
    return {
        "gens": generations, "pop":  population,
        "i25":  infected_vals[0], "i75":  infected_vals[1],
        "i10":  infected_vals[2], "i60":  infected_vals[3],
        "i50":  infected_vals[4], "i100": infected_vals[5],
    }


def model_predict(model, ckpt, feats):
    """
    Normalize a feature dict and run it through SISNet, then unscale outputs.

    Parameters
    ----------
    model : SISNet
    ckpt  : dict   — checkpoint dict containing beta/gamma min/max scalers
    feats : dict   — output of build_features_from_subsims()

    Returns
    -------
    beta_hat, gamma_hat, r0_hat : float
    """
    beta_min,  beta_max  = ckpt["beta_min"],  ckpt["beta_max"]
    gamma_min, gamma_max = ckpt["gamma_min"], ckpt["gamma_max"]

    gens = float(feats["gens"])
    pop  = float(feats["pop"])

    def prop(x):
        return x / (pop + 1e-6)

    X = np.array([[
        gens / 500.0,
        pop  / 20_000.0,
        prop(feats["i25"]), prop(feats["i75"]),
        prop(feats["i10"]), prop(feats["i60"]),
        prop(feats["i50"]), prop(feats["i100"]),
    ]], dtype=np.float32)

    with torch.no_grad():
        out = model(torch.tensor(X)).cpu().numpy()[0]

    beta_hat  = float(out[0] * (beta_max  - beta_min)  + beta_min)
    gamma_hat = float(out[1] * (gamma_max - gamma_min) + gamma_min)
    r0_hat    = beta_hat / gamma_hat if gamma_hat > 0 else float("inf")

    return beta_hat, gamma_hat, r0_hat


def summarize(arr, name):
    """Return mean, std, and 95% percentile CI for a 1-D array."""
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {
        "name":  name,
        "mean":  float(arr.mean()),
        "std":   float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p2_5":  float(lo),
        "p97_5": float(hi),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Parametric bootstrap for SISNet uncertainty quantification."
    )
    ap.add_argument("--beta",  type=float, required=True, help="True beta")
    ap.add_argument("--gamma", type=float, required=True, help="True gamma")
    ap.add_argument("--pop",   type=int,   required=True, help="Population size N")
    ap.add_argument("--gens",  type=int,   required=True, help="Total generations T")
    ap.add_argument("--i0",    type=int,   default=None,
                    help="Initial infected I0 (default: 5%% of pop, min 1)")
    ap.add_argument("--B",     type=int,   default=500,  help="Bootstrap replicates")
    ap.add_argument("--seed",  type=int,   default=123,  help="Base random seed")
    ap.add_argument("--ckpt",  default=MODEL_PATH,  help="Path to model checkpoint")
    ap.add_argument("--out",   default=OUTPUT_PATH, help="Output CSV path")
    args = ap.parse_args()

    i0 = args.i0 if args.i0 is not None else max(1, int(0.05 * args.pop))

    # Load checkpoint and model
    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = SISNet()
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    print(f"Parametric bootstrap: B={args.B}, beta={args.beta}, "
          f"gamma={args.gamma}, pop={args.pop}, gens={args.gens}, I0={i0}")

    rows = []
    beta_hats, gamma_hats, r0_hats = [], [], []

    for b in range(args.B):
        # Unique seed per replicate so each sub-simulation is independent
        feats = build_features_from_subsims(
            beta=args.beta, gamma=args.gamma,
            population=args.pop, i0=i0,
            generations=args.gens,
            seed=args.seed + b,
        )
        beta_hat, gamma_hat, r0_hat = model_predict(model, ckpt, feats)

        rows.append({
            "rep": b,
            "true_beta": args.beta, "true_gamma": args.gamma,
            "pop": args.pop, "gens": args.gens, "i0": i0,
            **{k: feats[k] for k in ["i25", "i75", "i10", "i60", "i50", "i100"]},
            "beta_hat": beta_hat, "gamma_hat": gamma_hat, "r0_hat": r0_hat,
        })
        beta_hats.append(beta_hat)
        gamma_hats.append(gamma_hat)
        r0_hats.append(r0_hat)

    # Save results to CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Print summary
    print("\n=== Bootstrap summary (percentile 95% CI) ===")
    for s in [summarize(beta_hats, "beta_hat"),
              summarize(gamma_hats, "gamma_hat"),
              summarize(r0_hats,    "r0_hat")]:
        print(
            f"{s['name']}: mean={s['mean']:.6f}, std={s['std']:.6f}, "
            f"CI95=[{s['p2_5']:.6f}, {s['p97_5']:.6f}]"
        )

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()