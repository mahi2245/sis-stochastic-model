#!/usr/bin/env python3
"""Minimal delete-d jackknife for SIS model"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class SISNet(nn.Module):
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


def train_model(X_train, y_train, X_val, y_val, epochs, lr):
    model = SISNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_state = None
    
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
            best_state = model.state_dict().copy()
    
    return best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training_data_with_time.csv")
    parser.add_argument("--delete", type=int, default=10, help="Delete d samples per iteration")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="jackknife_results.csv")
    args = parser.parse_args()
    
    # Load and preprocess
    df = pd.read_csv(args.data)
    N = len(df)
    
    generations = df["generations"].values.astype(np.float32)
    population = df["population"].values.astype(np.float32)
    infected_cols = ["infected_t25", "infected_t75", "infected_t10", 
                     "infected_t60", "infected_t50", "infected_t100"]
    infected = df[infected_cols].values.astype(np.float32)
    beta = df["beta"].values.astype(np.float32)
    gamma = df["gamma"].values.astype(np.float32)
    
    # Normalize
    X = np.concatenate([
        (generations / 500.0)[:, None],
        (population / 20000.0)[:, None],
        infected / (population[:, None] + 1e-6)
    ], axis=1).astype(np.float32)
    
    # Scale targets
    beta_min, beta_max = beta.min(), beta.max()
    gamma_min, gamma_max = gamma.min(), gamma.max()
    y = np.stack([
        (beta - beta_min) / (beta_max - beta_min),
        (gamma - gamma_min) / (gamma_max - gamma_min)
    ], axis=1).astype(np.float32)
    
    # Delete-d jackknife
    d = args.delete
    num_iter = N // d
    print(f"Delete-{d} jackknife: {num_iter} iterations on {N} samples")
    print(f"Training {num_iter} models with {args.epochs} epochs each...")
    
    results = []
    np.random.seed(42)
    indices = np.random.permutation(N)
    
    for i in range(num_iter):
        # Hold out d samples
        start_idx = i * d
        end_idx = min((i + 1) * d, N)
        held_out = indices[start_idx:end_idx]
        
        mask = np.ones(N, dtype=bool)
        mask[held_out] = False
        
        X_train_jack = X[mask]
        y_train_jack = y[mask]
        X_held = X[held_out]
        y_held = y[held_out]
        
        # Split train/val
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_jack, y_train_jack, test_size=0.2, random_state=42
        )
        
        # Train
        best_state = train_model(
            torch.tensor(X_tr), torch.tensor(y_tr),
            torch.tensor(X_val), torch.tensor(y_val),
            args.epochs, args.lr
        )
        
        # Predict on held-out
        model = SISNet()
        model.load_state_dict(best_state)
        model.eval()
        
        with torch.no_grad():
            y_pred = model(torch.tensor(X_held)).numpy()
        
        # Unscale
        beta_pred = y_pred[:, 0] * (beta_max - beta_min) + beta_min
        gamma_pred = y_pred[:, 1] * (gamma_max - gamma_min) + gamma_min
        r0_pred = beta_pred / np.maximum(gamma_pred, 1e-9)
        
        beta_true = beta[held_out]
        gamma_true = gamma[held_out]
        r0_true = beta_true / np.maximum(gamma_true, 1e-9)
        
        # Store
        for j, idx in enumerate(held_out):
            results.append({
                'iteration': i,
                'sample_idx': idx,
                'beta_true': beta_true[j],
                'beta_pred': beta_pred[j],
                'beta_error': beta_pred[j] - beta_true[j],
                'gamma_true': gamma_true[j],
                'gamma_pred': gamma_pred[j],
                'gamma_error': gamma_pred[j] - gamma_true[j],
                'r0_true': r0_true[j],
                'r0_pred': r0_pred[j],
                'r0_error': r0_pred[j] - r0_true[j],
            })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_iter}] completed")
    
    # Save and compute stats
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    
    # Corrected jackknife statistics
    print(f"\n{'='*70}")
    print("JACKKNIFE STATISTICS")
    print(f"{'='*70}")
    
    for param in ['beta', 'gamma', 'r0']:
        pred_col = f'{param}_pred'
        true_col = f'{param}_true'
        error_col = f'{param}_error'
        
        predictions = df_results[pred_col].values
        true_values = df_results[true_col].values
        errors = df_results[error_col].values
        
        # Prediction statistics
        mean_pred = predictions.mean()
        std_pred = predictions.std(ddof=1)
        mean_true = true_values.mean()
        
        # Error statistics
        bias = errors.mean()
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors**2).mean())
        
        print(f"\n{param.upper()}:")
        print(f"  Mean prediction:  {mean_pred:.6f}")
        print(f"  Prediction std:   {std_pred:.6f}")
        print(f"  Mean true value:  {mean_true:.6f}")
        print(f"  Bias (avg error): {bias:+.6f}")
        print(f"  MAE:              {mae:.6f}")
        print(f"  RMSE:             {rmse:.6f}")
    
    print(f"\n{'='*70}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()