import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F

# --------------------------------------------------------
# 1. MODEL WITH 8 INPUT FEATURES (6 infected timepoints)
# --------------------------------------------------------
class SISNet(nn.Module):
    def __init__(self):
        super(SISNet, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))


# --------------------------------------------------------
# 2. TRAINING SCRIPT
# --------------------------------------------------------
if __name__ == "__main__":

    print("Training model with corrected normalization...")

    df = pd.read_csv("training_data_with_time.csv")

    # unpack fields
    generations = df["generations"].values.astype(np.float32)
    population  = df["population"].values.astype(np.float32)

    # Load all 6 infected timepoints
    infected_cols = [
        "infected_t25", "infected_t75",
        "infected_t10", "infected_t60",
        "infected_t50", "infected_t100"
    ]
    infected = df[infected_cols].values.astype(np.float32)

    beta = df["beta"].values.astype(np.float32)
    gamma = df["gamma"].values.astype(np.float32)

    # ----------------------------------------------------
    #  NORMALIZATION FIX (THE MOST IMPORTANT PART)
    # ----------------------------------------------------
    gen_norm = generations / 500.0
    pop_norm = population / 20000.0

    # infected fraction: I(t)/N
    infected_norm = infected / (population[:, None] + 1e-6)

    # Concatenate features:
    # 8 inputs: [gen_norm, pop_norm, 6 infected proportions]
    X = np.concatenate([
        gen_norm[:, None],
        pop_norm[:, None],
        infected_norm
    ], axis=1).astype(np.float32)

    # ----------------------------------------------------
    # SCALE TARGETS (β, γ) to 0–1 for stable learning
    # ----------------------------------------------------
    beta_min, beta_max = beta.min(), beta.max()
    gamma_min, gamma_max = gamma.min(), gamma.max()

    beta_scaled  = (beta  - beta_min)  / (beta_max  - beta_min)
    gamma_scaled = (gamma - gamma_min) / (gamma_max - gamma_min)

    y = np.stack([beta_scaled, gamma_scaled], axis=1).astype(np.float32)

    # Convert to tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # ----------------------------------------------------
    # TRAIN
    # ----------------------------------------------------
    model = SISNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0008)

    for epoch in range(2500):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # ----------------------------------------------------
    # SAVE MODEL + TARGET SCALING VALUES
    # ----------------------------------------------------
    torch.save({
        "model_state": model.state_dict(),
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max
    }, "sis_model_with_time_fixed.pth")

    print("✅ Training complete.")
