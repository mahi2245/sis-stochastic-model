import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from simple import simulate_sis  # your stochastic simulator

# ----------------------------------------------------
# LOAD TRAINED MODEL
# ----------------------------------------------------
checkpoint = torch.load("sis_model_best.pth", map_location="cpu", weights_only=False)
model_state = checkpoint["model_state"]
beta_min, beta_max = checkpoint["beta_min"], checkpoint["beta_max"]
gamma_min, gamma_max = checkpoint["gamma_min"], checkpoint["gamma_max"]

from train import SISNet  # import your model class
model = SISNet()
model.load_state_dict(model_state)
model.eval()


# ----------------------------------------------------
# ODE SIS MODEL FOR TRAJECTORY PREDICTION
# ----------------------------------------------------
def sis_ode(t, I, beta, gamma, N):
    S = N - I
    return beta * I * (S / N) - gamma * I


def run_deterministic(beta, gamma, N, I0, generations):
    sol = solve_ivp(
        lambda t, y: sis_ode(t, y, beta, gamma, N),
        t_span=[0, generations],
        y0=[I0],
        t_eval=np.arange(generations + 1)
    )
    return sol.t, sol.y[0]


# ----------------------------------------------------
# NORMALIZATION HELPERS (same as your train.py)
# ----------------------------------------------------
def normalize_inputs(generations, population, infected):
    gen_norm = generations / 500.0
    pop_norm = population / 20000.0
    infected_norm = infected / (population + 1e-6)
    return np.concatenate([[gen_norm, pop_norm], infected_norm])


def unscale_params(beta_scaled, gamma_scaled):
    beta = beta_scaled * (beta_max - beta_min) + beta_min
    gamma = gamma_scaled * (gamma_max - gamma_min) + gamma_min
    return beta, gamma


# ----------------------------------------------------
# VERIFICATION LOOP (test on 5 random outbreaks)
# ----------------------------------------------------
NUM_TESTS = 5

for test_id in range(NUM_TESTS):
    print(f"\n=== Test outbreak #{test_id} ===")

    # 1. Generate synthetic test outbreak with unknown parameters
    N = np.random.randint(500, 20000)
    I0 = np.random.randint(1, max(2, N // 20))
    beta_true = np.random.uniform(0.05, 0.95)
    gamma_true = np.random.uniform(0.05, 0.5)
    generations = np.random.randint(50, 300)

    history, _ = simulate_sis(N, I0, beta_true, gamma_true, generations, x=0)
    infected_series = np.array([h.sum() for h in history])

    # extract model’s 6 timepoints
    idxs = [
        int(0.25 * generations),
        int(0.75 * generations),
        int(0.10 * generations),
        int(0.60 * generations),
        int(0.50 * generations),
        int(1.0  * generations) - 1
    ]
    infected_points = infected_series[idxs]

    # 2. Normalize input
    X = normalize_inputs(generations, N, infected_points).astype(np.float32)
    X_tensor = torch.tensor(X).unsqueeze(0)

    # 3. Predict scaled beta, gamma
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()[0]
    beta_pred, gamma_pred = unscale_params(pred_scaled[0], pred_scaled[1])

    print(f"True beta:   {beta_true:.4f} | Pred beta:   {beta_pred:.4f}")
    print(f"True gamma:  {gamma_true:.4f} | Pred gamma:  {gamma_pred:.4f}")
    print(f"True R0:     {beta_true/gamma_true:.3f} | Pred R0: {beta_pred/gamma_pred:.3f}")

    # 4. Simulate deterministic curve using predicted parameters
    t_pred, I_pred_curve = run_deterministic(beta_pred, gamma_pred, N, I0, generations)

    # 5. Plot comparison
    plt.figure(figsize=(8,5))
    plt.plot(infected_series, label="Stochastic True")
    plt.plot(t_pred, I_pred_curve, label="Pred Deterministic", linestyle="--")
    plt.title(f"Trajectory Comparison (Test #{test_id})")
    plt.xlabel("Generation")
    plt.ylabel("Infected")
    plt.legend()
    plt.tight_layout()
    plt.show()
