"""
train.py
--------
Define SIRNet and train it on synthetic SIR simulation data.

Usage (from project root):
    python src/train.py

Reads:
    data/training_data_with_time.csv   (produced by src/runX.py)

Writes:
    models/sir_model_best.pth          (best checkpoint by validation loss)

The checkpoint dict contains:
    model_state  : OrderedDict  — best model weights
    beta_min     : float        — min beta in training set (for unscaling)
    beta_max     : float        — max beta in training set (for unscaling)
    gamma_min    : float        — min gamma in training set (for unscaling)
    gamma_max    : float        — max gamma in training set (for unscaling)
    best_val_loss: float        — lowest MSE on the 20% validation split
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH  = "data/training_data_with_time.csv"
MODEL_PATH = "models/sir_model_best.pth"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SIRNet(nn.Module):
    """
    Feedforward network that maps 14 epidemic features to (beta, gamma).

    Input features (14):
        gen_norm        — generations / 500
        pop_norm        — population / 20000
        infected_t*     — 6 infected proportions (infected / population)
        recovered_t*    — 6 recovered proportions (recovered / population)

    Output (2):
        beta_scaled, gamma_scaled — both in [0, 1]; unscale using checkpoint
        min/max values to recover true beta and gamma.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(14, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))  # outputs in (0, 1)


# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # --- Unpack raw columns ---
    generations = df["generations"].values.astype(np.float32)
    population  = df["population"].values.astype(np.float32)

    infected_cols  = ["infected_t10",  "infected_t25",  "infected_t50",
                      "infected_t60",  "infected_t75",  "infected_t100"]
    recovered_cols = ["recovered_t10", "recovered_t25", "recovered_t50",
                      "recovered_t60", "recovered_t75", "recovered_t100"]

    infected  = df[infected_cols].values.astype(np.float32)
    recovered = df[recovered_cols].values.astype(np.float32)
    beta      = df["beta"].values.astype(np.float32)
    gamma     = df["gamma"].values.astype(np.float32)

    # --- Normalize features ---
    gen_norm      = generations / 500.0
    pop_norm      = population  / 20_000.0
    infected_norm  = infected  / (population[:, None] + 1e-6)  # proportion
    recovered_norm = recovered / (population[:, None] + 1e-6)  # proportion

    X = np.concatenate([
        gen_norm[:, None],
        pop_norm[:, None],
        infected_norm,
        recovered_norm,
    ], axis=1).astype(np.float32)

    # --- Scale targets to [0, 1] using dataset min/max ---
    beta_min,  beta_max  = beta.min(),  beta.max()
    gamma_min, gamma_max = gamma.min(), gamma.max()

    beta_scaled  = (beta  - beta_min)  / (beta_max  - beta_min)
    gamma_scaled = (gamma - gamma_min) / (gamma_max - gamma_min)
    y = np.stack([beta_scaled, gamma_scaled], axis=1).astype(np.float32)

    # --- Train / validation split (80 / 20) ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_val   = torch.tensor(X_val)
    y_val   = torch.tensor(y_val)

    # --- Training loop with best-model tracking ---
    model     = SIRNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0008)

    best_val_loss   = float("inf")
    best_state_dict = None

    print("Training SIRNet ...")
    for epoch in range(2500):
        # train step
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        if val_loss.item() < best_val_loss:
            best_val_loss   = val_loss.item()
            best_state_dict = model.state_dict()

        if epoch % 200 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {loss.item():.6f} | "
                f"Val Loss:   {val_loss.item():.6f}"
            )

    # --- Save best checkpoint ---
    torch.save(
        {
            "model_state":   best_state_dict,
            "beta_min":      beta_min,
            "beta_max":      beta_max,
            "gamma_min":     gamma_min,
            "gamma_max":     gamma_max,
            "best_val_loss": best_val_loss,
        },
        MODEL_PATH,
    )

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to {MODEL_PATH}")