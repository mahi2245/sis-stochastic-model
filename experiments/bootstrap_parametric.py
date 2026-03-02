#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import torch

from simple import simulate_sis  # uses your simulator

TIME_POINTS = [0.25, 0.75, 0.10, 0.60, 0.50, 1.0]  # matches runX.py

import torch.nn as nn
import torch.nn.functional as F

class SISNet(nn.Module):
    """
    Matches sis_model_best.pth:
      8 -> 32 -> 32 -> 16 -> 2
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
        # outputs are scaled in [0,1] in your predict.py unscale logic
        x = torch.sigmoid(self.fc4(x))
        return x


def build_features_from_subsims(beta, gamma, population, i0, generations, seed=None):
    """
    Mirrors runX.py: for each time point tp, run a fresh simulate_sis up to t=tp*generations
    and use the infected count from the final generation of that sub-simulation.
    """
    if seed is not None:
        np.random.seed(seed)

    infected_vals = []
    for tp in TIME_POINTS:
        t = max(1, int(tp * generations))
        hist_tp, _ = simulate_sis(
            n=population,
            m=i0,
            beta=beta,
            gamma=gamma,
            generations=t,
            x=0
        )
        infected_vals.append(int(np.sum(hist_tp[-1])))

    # Return dict for clarity
    return {
        "gens": generations,
        "pop": population,
        "i25": infected_vals[0],
        "i75": infected_vals[1],
        "i10": infected_vals[2],
        "i60": infected_vals[3],
        "i50": infected_vals[4],
        "i100": infected_vals[5],
    }


def model_predict(model, ckpt, feats):
    """
    Matches predict.py normalization + unscale.
    """
    beta_min  = ckpt["beta_min"]
    beta_max  = ckpt["beta_max"]
    gamma_min = ckpt["gamma_min"]
    gamma_max = ckpt["gamma_max"]

    gens = float(feats["gens"])
    pop  = float(feats["pop"])

    gen_norm = gens / 500.0
    pop_norm = pop / 20000.0

    i25_norm  = feats["i25"]  / (pop + 1e-6)
    i75_norm  = feats["i75"]  / (pop + 1e-6)
    i10_norm  = feats["i10"]  / (pop + 1e-6)
    i60_norm  = feats["i60"]  / (pop + 1e-6)
    i50_norm  = feats["i50"]  / (pop + 1e-6)
    i100_norm = feats["i100"] / (pop + 1e-6)

    X = np.array([[
        gen_norm,
        pop_norm,
        i25_norm,
        i75_norm,
        i10_norm,
        i60_norm,
        i50_norm,
        i100_norm
    ]], dtype=np.float32)

    with torch.no_grad():
        out = model(torch.tensor(X)).cpu().numpy()[0]

    beta_scaled, gamma_scaled = out
    beta_hat  = float(beta_scaled  * (beta_max  - beta_min)  + beta_min)
    gamma_hat = float(gamma_scaled * (gamma_max - gamma_min) + gamma_min)
    r0_hat = beta_hat / gamma_hat if gamma_hat > 0 else float("inf")
    return beta_hat, gamma_hat, r0_hat


def summarize(arr, name):
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {
        "name": name,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p2_5": float(lo),
        "p97_5": float(hi),
    }


def main():
    ap = argparse.ArgumentParser(description="Parametric bootstrap for SISNet predictions.")
    ap.add_argument("--ckpt", default="sis_model_best.pth", help="Path to checkpoint (e.g., sis_model_best.pth)")
    ap.add_argument("--beta", type=float, required=True, help="True beta to simulate under")
    ap.add_argument("--gamma", type=float, required=True, help="True gamma to simulate under")
    ap.add_argument("--pop", type=int, required=True, help="Population size N")
    ap.add_argument("--gens", type=int, required=True, help="Total generations T")
    ap.add_argument("--i0", type=int, default=None, help="Initial infected I0 (default: 5% of pop, min 1)")
    ap.add_argument("--B", type=int, default=500, help="Bootstrap replicates")
    ap.add_argument("--seed", type=int, default=123, help="Base random seed")
    ap.add_argument("--out", default="bootstrap_results.csv", help="Output CSV filename")
    args = ap.parse_args()

    i0 = args.i0 if args.i0 is not None else max(1, int(0.05 * args.pop))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)


    # Same pattern as predict.py: SISNet in train.py, load model_state, eval
    model = SISNet()
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()


    rows = []
    beta_hats, gamma_hats, r0_hats = [], [], []

    for b in range(args.B):
        # vary seed per replicate so subsims differ
        feats = build_features_from_subsims(
            beta=args.beta,
            gamma=args.gamma,
            population=args.pop,
            i0=i0,
            generations=args.gens,
            seed=args.seed + b
        )
        beta_hat, gamma_hat, r0_hat = model_predict(model, ckpt, feats)

        rows.append({
            "rep": b,
            "true_beta": args.beta,
            "true_gamma": args.gamma,
            "pop": args.pop,
            "gens": args.gens,
            "i0": i0,
            **{k: feats[k] for k in ["i25", "i75", "i10", "i60", "i50", "i100"]},
            "beta_hat": beta_hat,
            "gamma_hat": gamma_hat,
            "r0_hat": r0_hat,
        })

        beta_hats.append(beta_hat)
        gamma_hats.append(gamma_hat)
        r0_hats.append(r0_hat)

    # save CSV
    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # print summaries
    s_beta = summarize(beta_hats, "beta_hat")
    s_gamma = summarize(gamma_hats, "gamma_hat")
    s_r0 = summarize(r0_hats, "r0_hat")

    print("\n=== Bootstrap summary (percentile 95% CI) ===")
    for s in [s_beta, s_gamma, s_r0]:
        print(
            f"{s['name']}: mean={s['mean']:.6f}, std={s['std']:.6f}, "
            f"CI95=[{s['p2_5']:.6f}, {s['p97_5']:.6f}]"
        )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
