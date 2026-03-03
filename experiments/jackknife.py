#!/usr/bin/env python3
"""
jackknife.py
------------
Delete-d jackknife for SISNet uncertainty quantification.

Iteratively holds out d samples at a time, retrains SISNet on the
remaining data, and records predictions on the held-out set. This is
repeated across all N/d partitions. The resulting per-sample errors
can be used to estimate bias and variance of the SIS parameter estimates.

Usage (from project root):
    python experiments/jackknife.py [--delete 10] [--epochs 500]

Reads:
    data/training_data_with_time.csv

Writes:
    results/jackknife/jackknife_results.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH   = "data/training_data_with_time.csv"
OUTPUT_PATH = "results/jackknife/jackknife_results.csv"


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
        beta_scaled, gamma_scaled — both in (0, 1); unscale via dataset
        min/max values to recover true beta and gamma.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, X_val, y_val, epochs, lr):
    """
    Train a single SISNet and return the state dict with the best validation loss.

    Parameters
    ----------
    X_train, y_train : torch.Tensor  — jackknife training data (N-d samples)
    X_val,   y_val   : torch.Tensor  — validation split for early stopping
    epochs : int
    lr     : float

    Returns
    -------
    best_state : OrderedDict — model weights at the epoch of lowest val loss
    """
    model     = SISNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state    = model.state_dict().copy()

    return best_state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Delete-d jackknife for SISNet uncertainty quantification."
    )
    parser.add_argument("--data",   default=DATA_PATH,   help="Path to training CSV")
    parser.add_argument("--delete", type=int,   default=10,   help="Samples to hold out per iteration (d)")
    parser.add_argument("--epochs", type=int,   default=500,  help="Training epochs per model")
    parser.add_argument("--lr",     type=float, default=0.001,help="Learning rate")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    # --- Load and preprocess data ---
    df = pd.read_csv(args.data)
    N  = len(df)

    generations   = df["generations"].values.astype(np.float32)
    population    = df["population"].values.astype(np.float32)
    infected_cols = ["infected_t25", "infected_t75", "infected_t10",
                     "infected_t60", "infected_t50", "infected_t100"]
    infected      = df[infected_cols].values.astype(np.float32)
    beta          = df["beta"].values.astype(np.float32)
    gamma         = df["gamma"].values.astype(np.float32)

    # Feature matrix (8 columns)
    X = np.concatenate([
        (generations / 500.0)[:, None],
        (population  / 20_000.0)[:, None],
        infected / (population[:, None] + 1e-6),
    ], axis=1).astype(np.float32)

    # Scale targets to [0, 1]
    beta_min,  beta_max  = beta.min(),  beta.max()
    gamma_min, gamma_max = gamma.min(), gamma.max()

    y = np.stack([
        (beta  - beta_min)  / (beta_max  - beta_min),
        (gamma - gamma_min) / (gamma_max - gamma_min),
    ], axis=1).astype(np.float32)

    # --- Delete-d jackknife loop ---
    d        = args.delete
    num_iter = N // d
    print(f"Delete-{d} jackknife: {num_iter} iterations on {N} samples")
    print(f"Training {num_iter} models with {args.epochs} epochs each...")

    results = []
    np.random.seed(42)
    indices = np.random.permutation(N)  # random partition order

    for i in range(num_iter):
        # Hold out the next d samples from the shuffled index
        start_idx = i * d
        end_idx   = min((i + 1) * d, N)
        held_out  = indices[start_idx:end_idx]

        mask = np.ones(N, dtype=bool)
        mask[held_out] = False

        X_train_jack = X[mask]
        y_train_jack = y[mask]
        X_held       = X[held_out]

        # Inner train/val split for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_jack, y_train_jack, test_size=0.2, random_state=42
        )

        best_state = train_model(
            torch.tensor(X_tr), torch.tensor(y_tr),
            torch.tensor(X_val), torch.tensor(y_val),
            args.epochs, args.lr,
        )

        # Predict on held-out samples
        model = SISNet()
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            y_pred = model(torch.tensor(X_held)).numpy()

        # Unscale predictions
        beta_pred  = y_pred[:, 0] * (beta_max  - beta_min)  + beta_min
        gamma_pred = y_pred[:, 1] * (gamma_max - gamma_min) + gamma_min
        r0_pred    = beta_pred / np.maximum(gamma_pred, 1e-9)

        beta_true  = beta[held_out]
        gamma_true = gamma[held_out]
        r0_true    = beta_true / np.maximum(gamma_true, 1e-9)

        for j, idx in enumerate(held_out):
            results.append({
                "iteration":   i,
                "sample_idx":  idx,
                "beta_true":   beta_true[j],
                "beta_pred":   beta_pred[j],
                "beta_error":  beta_pred[j]  - beta_true[j],
                "gamma_true":  gamma_true[j],
                "gamma_pred":  gamma_pred[j],
                "gamma_error": gamma_pred[j] - gamma_true[j],
                "r0_true":     r0_true[j],
                "r0_pred":     r0_pred[j],
                "r0_error":    r0_pred[j]    - r0_true[j],
            })

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{num_iter}] completed")

    # --- Save results and print statistics ---
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)

    print(f"\n{'=' * 70}")
    print("JACKKNIFE STATISTICS")
    print(f"{'=' * 70}")

    for param in ["beta", "gamma", "r0"]:
        predictions  = df_results[f"{param}_pred"].values
        true_values  = df_results[f"{param}_true"].values
        errors       = df_results[f"{param}_error"].values

        bias = errors.mean()
        mae  = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())

        print(f"\n{param.upper()}:")
        print(f"  Mean prediction:  {predictions.mean():.6f}")
        print(f"  Prediction std:   {predictions.std(ddof=1):.6f}")
        print(f"  Mean true value:  {true_values.mean():.6f}")
        print(f"  Bias (avg error): {bias:+.6f}")
        print(f"  MAE:              {mae:.6f}")
        print(f"  RMSE:             {rmse:.6f}")

    print(f"\n{'=' * 70}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()