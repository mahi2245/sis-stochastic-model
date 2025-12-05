import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

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

    infected_cols = [
        "infected_t25", "infected_t75",
        "infected_t10", "infected_t60",
        "infected_t50", "infected_t100"
    ]
    infected = df[infected_cols].values.astype(np.float32)

    beta = df["beta"].values.astype(np.float32)
    gamma = df["gamma"].values.astype(np.float32)

    # normalization
    gen_norm = generations / 500.0
    pop_norm = population / 20000.0
    infected_norm = infected / (population[:, None] + 1e-6)

    # build feature matrix
    X = np.concatenate([
        gen_norm[:, None],
        pop_norm[:, None],
        infected_norm
    ], axis=1).astype(np.float32)

    # scale targets
    beta_min, beta_max = beta.min(), beta.max()
    gamma_min, gamma_max = gamma.min(), gamma.max()

    beta_scaled  = (beta  - beta_min)  / (beta_max  - beta_min)
    gamma_scaled = (gamma - gamma_min) / (gamma_max - gamma_min)
    y = np.stack([beta_scaled, gamma_scaled], axis=1).astype(np.float32)

    # ----------------------------------------------------
    #  TRAIN/VAL SPLIT (80 / 20)
    # ----------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_val   = torch.tensor(X_val)
    y_val   = torch.tensor(y_val)

    # ----------------------------------------------------
    # TRAINING LOOP WITH BEST-MODEL TRACKING
    # ----------------------------------------------------
    model = SISNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0008)

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(2500):
        # ---- TRAIN STEP ----
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # ---- VALIDATION LOSS ----
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        # track best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state_dict = model.state_dict()

        # print progress
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

    # ----------------------------------------------------
    # SAVE BEST MODEL ONLY
    # ----------------------------------------------------
    torch.save({
        "model_state": best_state_dict,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "best_val_loss": best_val_loss
    }, "sis_model_best.pth")

    print("Training complete.")
    print(f"Lowest validation loss: {best_val_loss:.6f}")
