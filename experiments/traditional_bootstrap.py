#!/usr/bin/env python3
"""Traditional (non-parametric) bootstrap for SIS model with train/test split"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class SISNet(nn.Module):
    """Neural network: 8 -> 32 -> 32 -> 16 -> 2"""
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
    """Train a single model and return best state"""
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
    parser = argparse.ArgumentParser(description="Traditional bootstrap for SIS model")
    parser.add_argument("--data", default="training_data_with_time_real.csv")
    parser.add_argument("--B", type=int, default=20, help="Number of bootstrap iterations")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs per model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="traditional_bootstrap_results.csv")
    args = parser.parse_args()
    
    print("="*70)
    print("TRADITIONAL BOOTSTRAP WITH TRAIN/TEST SPLIT")
    print("="*70)
    print(f"Bootstrap iterations: {args.B}")
    print(f"Epochs per model: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Test set size: {args.test_size*100:.0f}%")
    print("="*70)
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Filter out problematic rows
    df_clean = df[(df['infected_t100'] > 0) & ((df['beta']/df['gamma']) < 10)]
    N = len(df_clean)
    print(f"\nDataset: {N} samples after filtering")
    
    # Prepare features and targets
    generations = df_clean["generations"].values.astype(np.float32)
    population = df_clean["population"].values.astype(np.float32)
    infected_cols = ["infected_t25", "infected_t75", "infected_t10", 
                     "infected_t60", "infected_t50", "infected_t100"]
    infected = df_clean[infected_cols].values.astype(np.float32)
    beta = df_clean["beta"].values.astype(np.float32)
    gamma = df_clean["gamma"].values.astype(np.float32)
    
    # Normalize features
    X = np.concatenate([
        (generations / 500.0)[:, None],
        (population / 20000.0)[:, None],
        infected / (population[:, None] + 1e-6)
    ], axis=1).astype(np.float32)
    
    # Scale targets
    beta_min, beta_max = beta.min(), beta.max()
    gamma_min, gamma_max = gamma.min(), gamma.max()
    y = np.stack([
        (beta - beta_min) / (beta_max - beta_min + 1e-8),
        (gamma - gamma_min) / (gamma_max - gamma_min + 1e-8)
    ], axis=1).astype(np.float32)
    
    # Train/test split
    np.random.seed(args.seed)
    train_idx, test_idx = train_test_split(
        np.arange(N), test_size=args.test_size, random_state=args.seed
    )
    
    X_train_full = X[train_idx]
    y_train_full = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    beta_test_true = beta[test_idx]
    gamma_test_true = gamma[test_idx]
    
    print(f"Training set: {len(train_idx)} samples")
    print(f"Test set: {len(test_idx)} samples")
    
    # Store predictions for each test sample across bootstrap iterations
    n_test = len(test_idx)
    beta_preds = np.zeros((args.B, n_test))
    gamma_preds = np.zeros((args.B, n_test))
    r0_preds = np.zeros((args.B, n_test))
    
    print(f"\nTraining {args.B} bootstrap models...")
    print("-"*70)
    
    for b in range(args.B):
        # Bootstrap resample training data
        boot_idx = np.random.choice(len(train_idx), size=len(train_idx), replace=True)
        X_boot = X_train_full[boot_idx]
        y_boot = y_train_full[boot_idx]
        
        # Further split bootstrap sample into train/val for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_boot, y_boot, test_size=0.2, random_state=args.seed + b
        )
        
        # Train model
        best_state = train_model(
            torch.tensor(X_tr), torch.tensor(y_tr),
            torch.tensor(X_val), torch.tensor(y_val),
            args.epochs, args.lr
        )
        
        # Predict on test set
        model = SISNet()
        model.load_state_dict(best_state)
        model.eval()
        
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test)).numpy()
        
        # Unscale predictions
        beta_pred = y_pred[:, 0] * (beta_max - beta_min) + beta_min
        gamma_pred = y_pred[:, 1] * (gamma_max - gamma_min) + gamma_min
        r0_pred = beta_pred / np.maximum(gamma_pred, 1e-9)
        
        beta_preds[b] = beta_pred
        gamma_preds[b] = gamma_pred
        r0_preds[b] = r0_pred
        
        if (b + 1) % 5 == 0 or b == 0:
            print(f"  [{b+1}/{args.B}] completed")
    
    print("-"*70)
    print("Bootstrap training complete!")
    
    # Compute statistics for each test sample
    results = []
    for i in range(n_test):
        # Bootstrap statistics
        beta_mean = beta_preds[:, i].mean()
        beta_std = beta_preds[:, i].std(ddof=1)
        beta_ci = np.percentile(beta_preds[:, i], [2.5, 97.5])
        
        gamma_mean = gamma_preds[:, i].mean()
        gamma_std = gamma_preds[:, i].std(ddof=1)
        gamma_ci = np.percentile(gamma_preds[:, i], [2.5, 97.5])
        
        r0_mean = r0_preds[:, i].mean()
        r0_std = r0_preds[:, i].std(ddof=1)
        r0_ci = np.percentile(r0_preds[:, i], [2.5, 97.5])
        
        # True values
        beta_true = beta_test_true[i]
        gamma_true = gamma_test_true[i]
        r0_true = beta_true / gamma_true
        
        # Coverage
        beta_covered = beta_ci[0] <= beta_true <= beta_ci[1]
        gamma_covered = gamma_ci[0] <= gamma_true <= gamma_ci[1]
        r0_covered = r0_ci[0] <= r0_true <= r0_ci[1]
        
        results.append({
            'sample_idx': test_idx[i],
            'beta_true': beta_true,
            'beta_mean': beta_mean,
            'beta_std': beta_std,
            'beta_ci_low': beta_ci[0],
            'beta_ci_high': beta_ci[1],
            'beta_covered': beta_covered,
            'gamma_true': gamma_true,
            'gamma_mean': gamma_mean,
            'gamma_std': gamma_std,
            'gamma_ci_low': gamma_ci[0],
            'gamma_ci_high': gamma_ci[1],
            'gamma_covered': gamma_covered,
            'r0_true': r0_true,
            'r0_mean': r0_mean,
            'r0_std': r0_std,
            'r0_ci_low': r0_ci[0],
            'r0_ci_high': r0_ci[1],
            'r0_covered': r0_covered,
        })
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for param in ['beta', 'gamma', 'r0']:
        true_col = f'{param}_true'
        mean_col = f'{param}_mean'
        covered_col = f'{param}_covered'
        
        coverage = df_results[covered_col].mean() * 100
        bias = (df_results[mean_col] - df_results[true_col]).mean()
        mae = (df_results[mean_col] - df_results[true_col]).abs().mean()
        rmse = np.sqrt(((df_results[mean_col] - df_results[true_col])**2).mean())
        
        print(f"\n{param.upper()}:")
        print(f"  Coverage (95% CI): {coverage:.1f}%")
        print(f"  Bias:              {bias:+.6f}")
        print(f"  MAE:               {mae:.6f}")
        print(f"  RMSE:              {rmse:.6f}")
        print(f"  Mean CI width:     {(df_results[f'{param}_ci_high'] - df_results[f'{param}_ci_low']).mean():.6f}")
    
    print("\n" + "="*70)
    print(f"Saved: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()